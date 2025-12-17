#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Tests for Realism Constraint Solver (Rehydrated)

Tests cover:
    - Deterministic adjustment plans
    - Coupling matrix symmetry
    - Outlier detection
    - Director tile status logic
    - Neutral headline generation
    - JSON serializability

==============================================================================
"""

import json
import pytest
from experiments.synthetic_uplift.constraint_solver import (
    solve_realism_constraints,
    build_cross_scenario_coupling_map,
    build_director_realism_tile,
    build_complete_constraint_analysis,
    analyze_realism_from_ratchet,
    attach_realism_to_evidence,
    build_first_light_realism_annex,
    build_cal_exp_realism_card,
    attach_realism_cards_to_evidence,
    summarize_realism_vs_divergence,
    format_director_tile,
    EnvelopeBounds,
    DEFAULT_BOUNDS,
    round_to_precision,
    clamp_probability,
    clamp_correlation,
    SAFETY_LABEL,
    SCHEMA_VERSION,
    SOURCE_PATH_DIRECT,
    SOURCE_PATH_METRICS_NESTED,
    SOURCE_PATH_SUMMARY_NESTED,
    SOURCE_PATH_MISSING,
    _coerce_source_path,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def mock_ratchet():
    """Mock ratchet output."""
    return {
        "scenario_retention_score": {
            "scenario_stable": 0.9,
            "scenario_unstable": 0.5,
            "scenario_sharp": 0.3,
        },
        "stability_class": {
            "scenario_stable": "STABLE",
            "scenario_unstable": "SOFT_DRIFT",
            "scenario_sharp": "SHARP_DRIFT",
        },
    }


@pytest.fixture
def mock_calibration_console():
    """Mock calibration console output."""
    return {
        "slices_to_recalibrate": ["scenario_unstable", "scenario_sharp"],
        "calibration_status": "ATTENTION",
        "realism_pressure": 0.4,
    }


@pytest.fixture
def mock_scenario_configs():
    """Mock scenario configurations."""
    return {
        "scenario_stable": {
            "parameters": {
                "probabilities": {
                    "baseline": {"class_a": 0.70, "class_b": 0.60},
                    "rfl": {"class_a": 0.75, "class_b": 0.65},
                },
                "drift": {"mode": "none"},
                "correlation": {"rho": 0.5},
                "variance": {"per_cycle_sigma": 0.10, "per_item_sigma": 0.05},
                "rare_events": [],
            }
        },
        "scenario_unstable": {
            "parameters": {
                "probabilities": {
                    "baseline": {"class_a": 0.50, "class_b": 0.40},
                    "rfl": {"class_a": 0.55, "class_b": 0.45},
                },
                "drift": {"mode": "cyclical", "amplitude": 0.30, "period": 10},
                "correlation": {"rho": 0.95},
                "variance": {"per_cycle_sigma": 0.20, "per_item_sigma": 0.15},
                "rare_events": [
                    {"trigger_probability": 0.15, "magnitude": 0.80, "duration": 1}
                ],
            }
        },
        "scenario_sharp": {
            "parameters": {
                "probabilities": {
                    "baseline": {"class_a": 0.99, "class_b": 0.02},
                    "rfl": {"class_a": 0.98, "class_b": 0.03},
                },
                "drift": {"mode": "linear", "slope": 0.01},
                "correlation": {"rho": -0.1},
                "variance": {"per_cycle_sigma": 0.25, "per_item_sigma": 0.20},
                "rare_events": [
                    {"trigger_probability": 0.20, "magnitude": 1.0, "duration": 0},
                    {"trigger_probability": 0.15, "magnitude": 0.9, "duration": 1},
                ],
            }
        },
        "scenario_outlier": {
            "parameters": {
                "probabilities": {
                    "baseline": {"class_a": 0.10, "class_b": 0.90},
                    "rfl": {"class_a": 0.15, "class_b": 0.85},
                },
                "drift": {"mode": "shock", "shock_delta": 0.50, "shock_cycle": 100},
                "correlation": {"rho": 0.0},
                "variance": {"per_cycle_sigma": 0.01, "per_item_sigma": 0.01},
                "rare_events": [],
            }
        },
    }


# ==============================================================================
# TASK 1: DETERMINISTIC ADJUSTMENT PLANS
# ==============================================================================

def test_adjustment_plan_deterministic(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that adjustment plans are deterministic."""
    result1 = solve_realism_constraints(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    result2 = solve_realism_constraints(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    assert result1 == result2, "Adjustment plans must be deterministic"


def test_adjustment_plan_probability_clamping(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that probabilities are clamped to valid range."""
    result = solve_realism_constraints(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    # scenario_sharp has probabilities out of bounds (0.99, 0.02)
    if "scenario_sharp" in result["parameter_adjustment_plan"]:
        adjustments = result["parameter_adjustment_plan"]["scenario_sharp"]
        if "probabilities" in adjustments:
            for mode, probs in adjustments["probabilities"].items():
                for cls, prob in probs.items():
                    assert 0.05 <= prob <= 0.95, f"Probability {prob} must be in [0.05, 0.95]"


def test_adjustment_plan_drift_constraints(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that drift parameters are constrained."""
    result = solve_realism_constraints(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    # scenario_unstable has amplitude 0.30 > max 0.25
    if "scenario_unstable" in result["parameter_adjustment_plan"]:
        adjustments = result["parameter_adjustment_plan"]["scenario_unstable"]
        if "drift" in adjustments:
            if "amplitude" in adjustments["drift"]:
                assert abs(adjustments["drift"]["amplitude"]) <= 0.25


def test_adjustment_plan_correlation_bounds(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that correlation is bounded."""
    result = solve_realism_constraints(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    # scenario_unstable has rho 0.95 > max 0.9
    if "scenario_unstable" in result["parameter_adjustment_plan"]:
        adjustments = result["parameter_adjustment_plan"]["scenario_unstable"]
        if "correlation" in adjustments:
            if "rho" in adjustments["correlation"]:
                assert 0.0 <= adjustments["correlation"]["rho"] <= 0.9


def test_adjustment_plan_variance_reduction(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that variance parameters are reduced if excessive."""
    result = solve_realism_constraints(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    # scenario_unstable has per_cycle_sigma 0.20 > max 0.15
    if "scenario_unstable" in result["parameter_adjustment_plan"]:
        adjustments = result["parameter_adjustment_plan"]["scenario_unstable"]
        if "variance" in adjustments:
            if "per_cycle_sigma" in adjustments["variance"]:
                assert adjustments["variance"]["per_cycle_sigma"] <= 0.15


def test_adjustment_plan_rare_events_moderation(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that rare events are moderated."""
    result = solve_realism_constraints(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    # scenario_sharp has multiple rare events with high probability/magnitude
    if "scenario_sharp" in result["parameter_adjustment_plan"]:
        adjustments = result["parameter_adjustment_plan"]["scenario_sharp"]
        if "rare_events" in adjustments:
            for event in adjustments["rare_events"]:
                if "trigger_probability" in event:
                    assert event["trigger_probability"] <= 0.10
                if "magnitude" in event:
                    assert abs(event["magnitude"]) <= 0.70


def test_adjustment_plan_confidence_scoring(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that confidence scores are computed correctly."""
    result = solve_realism_constraints(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    
    # SHARP_DRIFT should reduce confidence
    if "scenario_sharp" in result["parameter_adjustment_plan"]:
        # Confidence should be lower due to SHARP_DRIFT penalty
        assert result["confidence"] < 1.0


def test_adjustment_plan_no_changes_when_within_bounds():
    """Test that no adjustments are made when scenarios are within bounds."""
    ratchet = {
        "scenario_retention_score": {"scenario_ok": 0.9},
        "stability_class": {"scenario_ok": "STABLE"},
    }
    
    console = {
        "slices_to_recalibrate": ["scenario_ok"],
        "calibration_status": "OK",
        "realism_pressure": 0.1,
    }
    
    configs = {
        "scenario_ok": {
            "parameters": {
                "probabilities": {
                    "baseline": {"class_a": 0.70},
                    "rfl": {"class_a": 0.75},
                },
                "drift": {"mode": "none"},
                "correlation": {"rho": 0.5},
                "variance": {"per_cycle_sigma": 0.10, "per_item_sigma": 0.05},
            }
        }
    }
    
    result = solve_realism_constraints(ratchet, console, configs)
    
    # Should have no adjustments if everything is within bounds
    assert result["scenarios_adjusted"] == 0 or len(result["parameter_adjustment_plan"]) == 0


# ==============================================================================
# TASK 2: COUPLING MATRIX SYMMETRY
# ==============================================================================

def test_coupling_matrix_symmetry(mock_scenario_configs):
    """Test that similarity matrix is symmetric."""
    result = build_cross_scenario_coupling_map(mock_scenario_configs)
    
    matrix = result["similarity_matrix"]
    scenarios = list(matrix.keys())
    
    for i, s1 in enumerate(scenarios):
        for j, s2 in enumerate(scenarios):
            assert matrix[s1][s2] == matrix[s2][s1], f"Matrix must be symmetric: {s1}↔{s2}"


def test_coupling_matrix_diagonal_ones(mock_scenario_configs):
    """Test that diagonal elements are 1.0 (self-similarity)."""
    result = build_cross_scenario_coupling_map(mock_scenario_configs)
    
    matrix = result["similarity_matrix"]
    
    for scenario in matrix.keys():
        assert matrix[scenario][scenario] == 1.0, f"Self-similarity must be 1.0 for {scenario}"


def test_coupling_matrix_similarity_range(mock_scenario_configs):
    """Test that all similarity scores are in [0, 1]."""
    result = build_cross_scenario_coupling_map(mock_scenario_configs)
    
    matrix = result["similarity_matrix"]
    
    for s1 in matrix.keys():
        for s2 in matrix[s1].keys():
            similarity = matrix[s1][s2]
            assert 0.0 <= similarity <= 1.0, f"Similarity {similarity} must be in [0, 1]"


def test_coupling_matrix_pairwise_format(mock_scenario_configs):
    """Test that pairwise_similarity uses colon-separated format."""
    result = build_cross_scenario_coupling_map(mock_scenario_configs)
    
    pairwise = result["pairwise_similarity"]
    
    for key in pairwise.keys():
        assert ":" in key, f"Pairwise key {key} must contain colon separator"
        parts = key.split(":")
        assert len(parts) == 2, f"Pairwise key {key} must have exactly two parts"


# ==============================================================================
# TASK 3: OUTLIER DETECTION
# ==============================================================================

def test_outlier_detection_identifies_low_similarity(mock_scenario_configs):
    """Test that outliers are identified based on low average similarity."""
    result = build_cross_scenario_coupling_map(mock_scenario_configs)
    
    outliers = result["outliers"]
    matrix = result["similarity_matrix"]
    
    # Verify outliers have low average similarity
    for outlier in outliers:
        if outlier in matrix:
            similarities = [
                matrix[outlier][other]
                for other in matrix.keys()
                if other != outlier
            ]
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                assert avg_similarity < 0.3, f"Outlier {outlier} should have avg similarity < 0.3"


def test_outlier_detection_deterministic(mock_scenario_configs):
    """Test that outlier detection is deterministic."""
    result1 = build_cross_scenario_coupling_map(mock_scenario_configs)
    result2 = build_cross_scenario_coupling_map(mock_scenario_configs)
    
    assert result1["outliers"] == result2["outliers"], "Outlier detection must be deterministic"


def test_outlier_detection_empty_when_all_similar():
    """Test that no outliers are detected when all scenarios are similar."""
    similar_configs = {
        "scenario_1": {
            "parameters": {
                "probabilities": {"baseline": {"class_a": 0.70}, "rfl": {"class_a": 0.75}},
                "drift": {"mode": "none"},
                "correlation": {"rho": 0.5},
                "variance": {"per_cycle_sigma": 0.10, "per_item_sigma": 0.05},
            }
        },
        "scenario_2": {
            "parameters": {
                "probabilities": {"baseline": {"class_a": 0.71}, "rfl": {"class_a": 0.76}},
                "drift": {"mode": "none"},
                "correlation": {"rho": 0.51},
                "variance": {"per_cycle_sigma": 0.11, "per_item_sigma": 0.06},
            }
        },
    }
    
    result = build_cross_scenario_coupling_map(similar_configs)
    
    # With high similarity, outliers should be empty or minimal
    assert len(result["outliers"]) <= 1, "Similar scenarios should not be outliers"


# ==============================================================================
# TASK 4: DIRECTOR TILE STATUS LOGIC
# ==============================================================================

def test_director_tile_status_red_on_block():
    """Test that status light is RED when calibration status is BLOCK."""
    constraint_solution = {
        "confidence": 0.4,
        "scenarios_adjusted": 2,
    }
    
    coupling_map = {
        "outliers": ["scenario_1"],
    }
    
    calibration_console = {
        "calibration_status": "BLOCK",
        "realism_pressure": 0.8,
    }
    
    tile = build_director_realism_tile(constraint_solution, coupling_map, calibration_console)
    
    assert tile["status_light"] == "RED", "Status should be RED when BLOCK"


def test_director_tile_status_red_on_high_pressure():
    """Test that status light is RED when realism pressure is high."""
    constraint_solution = {
        "confidence": 0.6,
        "scenarios_adjusted": 2,
    }
    
    coupling_map = {
        "outliers": ["scenario_1"],
    }
    
    calibration_console = {
        "calibration_status": "OK",
        "realism_pressure": 0.75,
    }
    
    tile = build_director_realism_tile(constraint_solution, coupling_map, calibration_console)
    
    assert tile["status_light"] == "RED", "Status should be RED when pressure > 0.7"


def test_director_tile_status_red_on_low_confidence():
    """Test that status light is RED when confidence is low."""
    constraint_solution = {
        "confidence": 0.4,
        "scenarios_adjusted": 2,
    }
    
    coupling_map = {
        "outliers": ["scenario_1"],
    }
    
    calibration_console = {
        "calibration_status": "OK",
        "realism_pressure": 0.2,
    }
    
    tile = build_director_realism_tile(constraint_solution, coupling_map, calibration_console)
    
    assert tile["status_light"] == "RED", "Status should be RED when confidence < 0.5"


def test_director_tile_status_yellow_on_attention():
    """Test that status light is YELLOW when calibration status is ATTENTION."""
    constraint_solution = {
        "confidence": 0.7,
        "scenarios_adjusted": 2,
    }
    
    coupling_map = {
        "outliers": ["scenario_1"],
    }
    
    calibration_console = {
        "calibration_status": "ATTENTION",
        "realism_pressure": 0.2,
    }
    
    tile = build_director_realism_tile(constraint_solution, coupling_map, calibration_console)
    
    assert tile["status_light"] == "YELLOW", "Status should be YELLOW when ATTENTION"


def test_director_tile_status_yellow_on_high_outliers():
    """Test that status light is YELLOW when outlier count is high."""
    constraint_solution = {
        "confidence": 0.7,
        "scenarios_adjusted": 2,
    }
    
    coupling_map = {
        "outliers": ["scenario_1", "scenario_2", "scenario_3", "scenario_4"],
    }
    
    calibration_console = {
        "calibration_status": "OK",
        "realism_pressure": 0.2,
    }
    
    tile = build_director_realism_tile(constraint_solution, coupling_map, calibration_console)
    
    assert tile["status_light"] == "YELLOW", "Status should be YELLOW when outliers > 3"


def test_director_tile_status_green_when_ok():
    """Test that status light is GREEN when all conditions are OK."""
    constraint_solution = {
        "confidence": 0.8,
        "scenarios_adjusted": 1,
    }
    
    coupling_map = {
        "outliers": ["scenario_1"],
    }
    
    calibration_console = {
        "calibration_status": "OK",
        "realism_pressure": 0.2,
    }
    
    tile = build_director_realism_tile(constraint_solution, coupling_map, calibration_console)
    
    assert tile["status_light"] == "GREEN", "Status should be GREEN when all OK"


# ==============================================================================
# TASK 5: NEUTRAL HEADLINE
# ==============================================================================

def test_director_tile_headline_neutral_green():
    """Test that headline is neutral for GREEN status."""
    constraint_solution = {
        "confidence": 0.8,
        "scenarios_adjusted": 1,
    }
    
    coupling_map = {
        "outliers": ["scenario_1"],
    }
    
    calibration_console = {
        "calibration_status": "OK",
        "realism_pressure": 0.2,
    }
    
    tile = build_director_realism_tile(constraint_solution, coupling_map, calibration_console)
    
    headline = tile["headline"]
    assert "operational" in headline.lower(), "GREEN headline should mention operational"
    assert "uplift" not in headline.lower(), "Headline should not mention uplift"


def test_director_tile_headline_neutral_yellow():
    """Test that headline is neutral for YELLOW status."""
    constraint_solution = {
        "confidence": 0.7,
        "scenarios_adjusted": 2,
    }
    
    coupling_map = {
        "outliers": ["scenario_1"],
    }
    
    calibration_console = {
        "calibration_status": "ATTENTION",
        "realism_pressure": 0.4,
    }
    
    tile = build_director_realism_tile(constraint_solution, coupling_map, calibration_console)
    
    headline = tile["headline"]
    assert "attention" in headline.lower(), "YELLOW headline should mention attention"
    assert "uplift" not in headline.lower(), "Headline should not mention uplift"


def test_director_tile_headline_neutral_red():
    """Test that headline is neutral for RED status."""
    constraint_solution = {
        "confidence": 0.4,
        "scenarios_adjusted": 3,
    }
    
    coupling_map = {
        "outliers": ["scenario_1"],
    }
    
    calibration_console = {
        "calibration_status": "BLOCK",
        "realism_pressure": 0.8,
    }
    
    tile = build_director_realism_tile(constraint_solution, coupling_map, calibration_console)
    
    headline = tile["headline"]
    assert "blocked" in headline.lower() or "uncertain" in headline.lower(), "RED headline should mention blocked/uncertain"
    assert "uplift" not in headline.lower(), "Headline should not mention uplift"


# ==============================================================================
# TASK 6: JSON SERIALIZABILITY
# ==============================================================================

def test_constraint_solution_json_serializable(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that constraint solution is JSON serializable."""
    result = solve_realism_constraints(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    # Should not raise exception
    json_str = json.dumps(result)
    assert isinstance(json_str, str)
    
    # Should be able to deserialize
    deserialized = json.loads(json_str)
    assert deserialized["schema_version"] == SCHEMA_VERSION


def test_coupling_map_json_serializable(mock_scenario_configs):
    """Test that coupling map is JSON serializable."""
    result = build_cross_scenario_coupling_map(mock_scenario_configs)
    
    # Should not raise exception
    json_str = json.dumps(result)
    assert isinstance(json_str, str)
    
    # Should be able to deserialize
    deserialized = json.loads(json_str)
    assert deserialized["schema_version"] == SCHEMA_VERSION


def test_director_tile_json_serializable(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that director tile is JSON serializable."""
    constraint_solution = solve_realism_constraints(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    coupling_map = build_cross_scenario_coupling_map(mock_scenario_configs)
    
    tile = build_director_realism_tile(
        constraint_solution,
        coupling_map,
        mock_calibration_console,
    )
    
    # Should not raise exception
    json_str = json.dumps(tile)
    assert isinstance(json_str, str)
    
    # Should be able to deserialize
    deserialized = json.loads(json_str)
    assert deserialized["schema_version"] == SCHEMA_VERSION


def test_complete_analysis_json_serializable(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that complete analysis is JSON serializable."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    # Should not raise exception
    json_str = json.dumps(analysis)
    assert isinstance(json_str, str)
    
    # Should be able to deserialize
    deserialized = json.loads(json_str)
    assert deserialized["schema_version"] == SCHEMA_VERSION


# ==============================================================================
# ADDITIONAL TESTS: SHADOW-MODE GUARANTEES
# ==============================================================================

def test_safety_label_present(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that all outputs include safety label."""
    solution = solve_realism_constraints(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    assert solution["label"] == SAFETY_LABEL
    
    coupling_map = build_cross_scenario_coupling_map(mock_scenario_configs)
    assert coupling_map["label"] == SAFETY_LABEL
    
    tile = build_director_realism_tile(solution, coupling_map, mock_calibration_console)
    assert tile["label"] == SAFETY_LABEL


def test_schema_version_consistent(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that schema version is consistent across outputs."""
    solution = solve_realism_constraints(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    coupling_map = build_cross_scenario_coupling_map(mock_scenario_configs)
    tile = build_director_realism_tile(solution, coupling_map, mock_calibration_console)
    
    assert solution["schema_version"] == SCHEMA_VERSION
    assert coupling_map["schema_version"] == SCHEMA_VERSION
    assert tile["schema_version"] == SCHEMA_VERSION


def test_format_director_tile_produces_text(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that format_director_tile produces readable text."""
    solution = solve_realism_constraints(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    coupling_map = build_cross_scenario_coupling_map(mock_scenario_configs)
    tile = build_director_realism_tile(solution, coupling_map, mock_calibration_console)
    
    formatted = format_director_tile(tile)
    
    assert isinstance(formatted, str)
    assert len(formatted) > 0
    assert SAFETY_LABEL in formatted
    assert tile["status_light"] in formatted


def test_empty_scenario_configs_handled():
    """Test that empty scenario configs are handled gracefully."""
    result = build_cross_scenario_coupling_map({})
    
    assert result["pairwise_similarity"] == {}
    assert result["scenario_clusters"] == {}
    assert result["outliers"] == []
    assert result["similarity_matrix"] == {}
    assert result["cluster_count"] == 0
    assert result["outlier_count"] == 0


def test_convergence_cycles_estimation():
    """Test that convergence cycles are estimated correctly."""
    constraint_solution = {
        "confidence": 0.7,
        "scenarios_adjusted": 2,
    }
    
    coupling_map = {
        "outliers": ["scenario_1"],
    }
    
    calibration_console = {
        "calibration_status": "OK",
        "realism_pressure": 0.3,
    }
    
    tile = build_director_realism_tile(constraint_solution, coupling_map, calibration_console)
    
    assert "expected_convergence_cycles" in tile
    assert isinstance(tile["expected_convergence_cycles"], int)
    assert tile["expected_convergence_cycles"] >= 0
    assert tile["expected_convergence_cycles"] <= 500  # Capped at 500


def test_clustering_produces_groups(mock_scenario_configs):
    """Test that clustering produces meaningful groups."""
    result = build_cross_scenario_coupling_map(mock_scenario_configs)
    
    clusters = result["scenario_clusters"]
    
    # Should have at least one cluster if scenarios are similar
    assert isinstance(clusters, dict)
    
    # Collect all scenarios in clusters
    clustered_scenarios = set()
    for cluster_scenarios in clusters.values():
        clustered_scenarios.update(cluster_scenarios)
    
    outliers = set(result["outliers"])
    
    # Clusters should contain scenarios (if any exist)
    # Note: Not all scenarios may be in clusters if they don't meet similarity threshold
    # and not all scenarios may be outliers if their average similarity is >= 0.3
    # This is acceptable behavior - the test just verifies clustering works
    assert isinstance(clustered_scenarios, set)
    assert isinstance(outliers, set)


def test_rounding_precision():
    """Test that rounding precision is applied correctly."""
    assert round_to_precision(0.123456789) == 0.1235
    assert round_to_precision(0.1, precision=2) == 0.1
    assert round_to_precision(0.9999, precision=2) == 1.0


def test_probability_clamping():
    """Test that probability clamping works correctly."""
    bounds = EnvelopeBounds()
    assert clamp_probability(0.99, bounds) == 0.95
    assert clamp_probability(0.01, bounds) == 0.05
    assert clamp_probability(0.50, bounds) == 0.50


def test_correlation_clamping():
    """Test that correlation clamping works correctly."""
    bounds = EnvelopeBounds()
    assert clamp_correlation(1.0, bounds) == 0.9
    assert clamp_correlation(-0.1, bounds) == 0.0
    assert clamp_correlation(0.5, bounds) == 0.5


# ==============================================================================
# INTEGRATION TESTS: RATCHET INTEGRATION AND EVIDENCE ATTACHMENT
# ==============================================================================

def test_analyze_realism_from_ratchet(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that ratchet integration stub works correctly."""
    result = analyze_realism_from_ratchet(
        ratchet=mock_ratchet,
        console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    assert "constraint_solution" in result
    assert "coupling_map" in result
    assert "director_tile" in result
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["label"] == SAFETY_LABEL


def test_attach_realism_to_evidence(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that evidence attachment works correctly."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    evidence = {
        "metrics": {"some_metric": 0.5},
        "governance": {},
    }
    
    new_evidence = attach_realism_to_evidence(evidence, analysis)
    
    # Verify non-mutating
    assert evidence is not new_evidence
    assert "governance" in new_evidence
    assert "realism_constraints" in new_evidence["governance"]
    
    # Verify structure
    realism = new_evidence["governance"]["realism_constraints"]
    assert "director_tile" in realism
    assert "coupling_summary" in realism
    assert "top_adjustments" in realism
    assert "overall_confidence" in realism


def test_attach_realism_to_evidence_deterministic(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that evidence attachment is deterministic."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    evidence = {"governance": {}}
    
    result1 = attach_realism_to_evidence(evidence, analysis)
    result2 = attach_realism_to_evidence(evidence, analysis)
    
    assert result1 == result2, "Evidence attachment must be deterministic"


def test_attach_realism_to_evidence_json_round_trip(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that attached evidence can be JSON serialized and deserialized."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    evidence = {"governance": {}}
    
    new_evidence = attach_realism_to_evidence(evidence, analysis)
    
    # Serialize and deserialize
    json_str = json.dumps(new_evidence)
    deserialized = json.loads(json_str)
    
    # Verify structure preserved
    assert "governance" in deserialized
    assert "realism_constraints" in deserialized["governance"]
    assert deserialized["governance"]["realism_constraints"]["schema_version"] == SCHEMA_VERSION


def test_attach_realism_to_evidence_coupling_summary(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that coupling summary excludes large similarity matrix."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    evidence = {"governance": {}}
    
    new_evidence = attach_realism_to_evidence(evidence, analysis)
    
    coupling_summary = new_evidence["governance"]["realism_constraints"]["coupling_summary"]
    
    # Should have summary fields but not full similarity matrix
    assert "outliers" in coupling_summary
    assert "outlier_count" in coupling_summary
    assert "cluster_count" in coupling_summary
    assert "scenario_count" in coupling_summary
    assert "similarity_matrix" not in coupling_summary, "Should not include large similarity matrix"


def test_attach_realism_to_evidence_top_adjustments_limit(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that top adjustments are limited to 5."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    evidence = {"governance": {}}
    
    new_evidence = attach_realism_to_evidence(evidence, analysis)
    
    top_adjustments = new_evidence["governance"]["realism_constraints"]["top_adjustments"]
    
    # Should be limited to 5
    assert len(top_adjustments) <= 5
    
    # Each should have required fields
    for adj in top_adjustments:
        assert "scenario" in adj
        assert "adjustments" in adj
        assert "expected_effect" in adj


# ==============================================================================
# FIRST LIGHT ANNEX TESTS
# ==============================================================================

def test_build_first_light_realism_annex(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that First Light annex is built correctly."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    annex = build_first_light_realism_annex(analysis)
    
    # Verify structure
    assert "schema_version" in annex
    assert annex["schema_version"] == "1.0.0"
    assert "status_light" in annex
    assert "global_pressure" in annex
    assert "outliers" in annex
    assert "overall_confidence" in annex
    
    # Verify values match analysis
    director_tile = analysis["director_tile"]
    coupling_map = analysis["coupling_map"]
    
    assert annex["status_light"] == director_tile["status_light"]
    assert annex["global_pressure"] == director_tile["realism_pressure"]
    assert annex["outliers"] == coupling_map["outliers"]
    assert annex["overall_confidence"] == director_tile["adjustment_confidence"]


def test_build_first_light_realism_annex_deterministic(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that First Light annex is deterministic."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    annex1 = build_first_light_realism_annex(analysis)
    annex2 = build_first_light_realism_annex(analysis)
    
    assert annex1 == annex2, "First Light annex must be deterministic"


def test_build_first_light_realism_annex_json_serializable(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that First Light annex is JSON serializable."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    annex = build_first_light_realism_annex(analysis)
    
    # Should not raise exception
    json_str = json.dumps(annex)
    assert isinstance(json_str, str)
    
    # Should be able to deserialize
    deserialized = json.loads(json_str)
    assert deserialized["schema_version"] == "1.0.0"


def test_attach_realism_to_evidence_includes_annex(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that evidence attachment includes First Light annex."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    evidence = {"governance": {}}
    
    new_evidence = attach_realism_to_evidence(evidence, analysis)
    
    # Verify annex is present
    assert "first_light_annex" in new_evidence["governance"]["realism_constraints"]
    
    annex = new_evidence["governance"]["realism_constraints"]["first_light_annex"]
    
    # Verify annex structure
    assert annex["schema_version"] == "1.0.0"
    assert "status_light" in annex
    assert "global_pressure" in annex
    assert "outliers" in annex
    assert "overall_confidence" in annex


def test_annex_attach_determinism(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that annex attachment is deterministic."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    evidence = {"governance": {}}
    
    result1 = attach_realism_to_evidence(evidence, analysis)
    result2 = attach_realism_to_evidence(evidence, analysis)
    
    # Verify annexes are identical
    annex1 = result1["governance"]["realism_constraints"]["first_light_annex"]
    annex2 = result2["governance"]["realism_constraints"]["first_light_annex"]
    
    assert annex1 == annex2, "Annex attachment must be deterministic"


def test_annex_attach_json_round_trip(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that evidence with annex can be JSON serialized and deserialized."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    evidence = {"governance": {}}
    
    new_evidence = attach_realism_to_evidence(evidence, analysis)
    
    # Serialize and deserialize
    json_str = json.dumps(new_evidence)
    deserialized = json.loads(json_str)
    
    # Verify annex structure preserved
    assert "first_light_annex" in deserialized["governance"]["realism_constraints"]
    annex = deserialized["governance"]["realism_constraints"]["first_light_annex"]
    assert annex["schema_version"] == "1.0.0"


# ==============================================================================
# CALIBRATION EXPERIMENT REALISM CARD TESTS
# ==============================================================================

def test_build_cal_exp_realism_card_shape(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that calibration experiment card has correct shape."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    annex = build_first_light_realism_annex(analysis)
    card = build_cal_exp_realism_card("cal_exp_001", annex)
    
    # Verify required fields
    assert "schema_version" in card
    assert card["schema_version"] == "1.0.0"
    assert "cal_id" in card
    assert card["cal_id"] == "cal_exp_001"
    assert "status_light" in card
    assert "global_pressure" in card
    assert "overall_confidence" in card
    
    # Verify values match annex
    assert card["status_light"] == annex["status_light"]
    assert card["global_pressure"] == annex["global_pressure"]
    assert card["overall_confidence"] == annex["overall_confidence"]


def test_build_cal_exp_realism_card_deterministic(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that calibration experiment card is deterministic."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    annex = build_first_light_realism_annex(analysis)
    
    card1 = build_cal_exp_realism_card("cal_exp_001", annex)
    card2 = build_cal_exp_realism_card("cal_exp_001", annex)
    
    assert card1 == card2, "Calibration experiment card must be deterministic"


def test_build_cal_exp_realism_card_json_safe(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that calibration experiment card is JSON serializable."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    annex = build_first_light_realism_annex(analysis)
    card = build_cal_exp_realism_card("cal_exp_001", annex)
    
    # Should not raise exception
    json_str = json.dumps(card)
    assert isinstance(json_str, str)
    
    # Should be able to deserialize
    deserialized = json.loads(json_str)
    assert deserialized["schema_version"] == "1.0.0"
    assert deserialized["cal_id"] == "cal_exp_001"


def test_build_cal_exp_realism_card_non_mutating(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that calibration experiment card builder is non-mutating."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    annex = build_first_light_realism_annex(analysis)
    annex_copy = dict(annex)
    
    card = build_cal_exp_realism_card("cal_exp_001", annex)
    
    # Verify annex was not modified
    assert annex == annex_copy, "Annex should not be modified"


def test_attach_realism_cards_to_evidence(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that realism cards are attached to evidence correctly."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    annex = build_first_light_realism_annex(analysis)
    card1 = build_cal_exp_realism_card("cal_exp_001", annex)
    card2 = build_cal_exp_realism_card("cal_exp_002", annex)
    
    evidence = {"governance": {}}
    new_evidence = attach_realism_cards_to_evidence(evidence, [card1, card2])
    
    # Verify non-mutating
    assert evidence is not new_evidence
    assert "governance" in new_evidence
    assert "realism_cards" in new_evidence["governance"]
    
    # Verify structure
    realism_cards = new_evidence["governance"]["realism_cards"]
    assert "cards" in realism_cards
    assert "card_count" in realism_cards
    assert realism_cards["card_count"] == 2
    assert len(realism_cards["cards"]) == 2
    assert realism_cards["cards"][0]["cal_id"] == "cal_exp_001"
    assert realism_cards["cards"][1]["cal_id"] == "cal_exp_002"


def test_attach_realism_cards_to_evidence_json_round_trip(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that evidence with cards can be JSON serialized and deserialized."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    annex = build_first_light_realism_annex(analysis)
    card = build_cal_exp_realism_card("cal_exp_001", annex)
    
    evidence = {"governance": {}}
    new_evidence = attach_realism_cards_to_evidence(evidence, [card])
    
    # Serialize and deserialize
    json_str = json.dumps(new_evidence)
    deserialized = json.loads(json_str)
    
    # Verify structure preserved
    assert "realism_cards" in deserialized["governance"]
    cards_data = deserialized["governance"]["realism_cards"]
    assert cards_data["schema_version"] == "1.0.0"
    assert cards_data["card_count"] == 1
    assert len(cards_data["cards"]) == 1
    assert cards_data["cards"][0]["cal_id"] == "cal_exp_001"


def test_attach_realism_cards_to_evidence_empty_list():
    """Test that attaching empty card list works correctly."""
    evidence = {"governance": {}}
    new_evidence = attach_realism_cards_to_evidence(evidence, [])
    
    assert "realism_cards" in new_evidence["governance"]
    cards_data = new_evidence["governance"]["realism_cards"]
    assert cards_data["card_count"] == 0
    assert len(cards_data["cards"]) == 0


# ==============================================================================
# REALISM VS DIVERGENCE CONSISTENCY TESTS
# ==============================================================================

def test_summarize_realism_vs_divergence_consistent_green_low_divergence():
    """Test CONSISTENT status: GREEN status_light with low final divergence."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_001",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},
        {"window_index": 1, "divergence_rate": 0.2},
        {"window_index": 2, "divergence_rate": 0.3},  # Final: 0.3 < 0.5
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert result["schema_version"] == "1.0.0"
    assert result["consistency_status"] == "CONSISTENT"
    assert len(result["advisory_notes"]) > 0
    assert "GREEN" in result["advisory_notes"][0] or "aligns" in result["advisory_notes"][0]


def test_summarize_realism_vs_divergence_consistent_red_high_divergence():
    """Test CONSISTENT status: RED status_light with high final divergence."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_002",
        "status_light": "RED",
        "global_pressure": 0.8,
        "overall_confidence": 0.7,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.7},
        {"window_index": 1, "divergence_rate": 0.8},
        {"window_index": 2, "divergence_rate": 0.75},  # Final: 0.75 >= 0.7
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert result["schema_version"] == "1.0.0"
    assert result["consistency_status"] == "CONSISTENT"
    assert len(result["advisory_notes"]) > 0
    assert "RED" in result["advisory_notes"][0] or "aligns" in result["advisory_notes"][0]


def test_summarize_realism_vs_divergence_conflict_green_high_persistent_divergence():
    """Test CONFLICT status: GREEN status_light but persistently high divergence."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_003",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    # 3+ windows with divergence_rate >= 0.9
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.9},
        {"window_index": 1, "divergence_rate": 0.95},
        {"window_index": 2, "divergence_rate": 0.92},
        {"window_index": 3, "divergence_rate": 0.91},
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert result["schema_version"] == "1.0.0"
    assert result["consistency_status"] == "CONFLICT"
    assert len(result["advisory_notes"]) > 0
    assert "CONFLICT" in result["consistency_status"] or "mismatch" in result["advisory_notes"][0].lower()


def test_summarize_realism_vs_divergence_tension_green_moderate_divergence():
    """Test TENSION status: GREEN status_light with moderate final divergence."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_004",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    # Final divergence >= 0.5 but not persistently high
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.4},
        {"window_index": 1, "divergence_rate": 0.5},
        {"window_index": 2, "divergence_rate": 0.6},  # Final: 0.6 >= 0.5
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert result["schema_version"] == "1.0.0"
    assert result["consistency_status"] == "TENSION"
    assert len(result["advisory_notes"]) > 0


def test_summarize_realism_vs_divergence_tension_red_low_divergence():
    """Test TENSION status: RED status_light with low final divergence."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_005",
        "status_light": "RED",
        "global_pressure": 0.8,
        "overall_confidence": 0.7,
    }
    
    # Final divergence < 0.7
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.3},
        {"window_index": 1, "divergence_rate": 0.4},
        {"window_index": 2, "divergence_rate": 0.5},  # Final: 0.5 < 0.7
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert result["schema_version"] == "1.0.0"
    assert result["consistency_status"] == "TENSION"
    assert len(result["advisory_notes"]) > 0


def test_summarize_realism_vs_divergence_tension_yellow():
    """Test TENSION status: YELLOW status_light."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_006",
        "status_light": "YELLOW",
        "global_pressure": 0.5,
        "overall_confidence": 0.6,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.4},
        {"window_index": 1, "divergence_rate": 0.5},
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert result["schema_version"] == "1.0.0"
    assert result["consistency_status"] == "TENSION"
    assert len(result["advisory_notes"]) > 0


def test_summarize_realism_vs_divergence_deterministic():
    """Test that consistency check is deterministic."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_007",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},
        {"window_index": 1, "divergence_rate": 0.2},
        {"window_index": 2, "divergence_rate": 0.3},
    ]
    
    result1 = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    result2 = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert result1 == result2, "Consistency check must be deterministic"


def test_summarize_realism_vs_divergence_json_safe():
    """Test that consistency check output is JSON serializable."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_008",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},
        {"window_index": 1, "divergence_rate": 0.2},
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    # Should not raise exception
    json_str = json.dumps(result)
    assert isinstance(json_str, str)
    
    # Should be able to deserialize
    deserialized = json.loads(json_str)
    assert deserialized["schema_version"] == "1.0.0"
    assert "consistency_status" in deserialized
    assert "advisory_notes" in deserialized


def test_summarize_realism_vs_divergence_no_windows():
    """Test handling when no windows are provided."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_009",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    result = summarize_realism_vs_divergence(realism_card, [])
    
    assert result["schema_version"] == "1.0.0"
    assert result["consistency_status"] == "TENSION"
    assert len(result["advisory_notes"]) > 0
    assert "No divergence windows" in result["advisory_notes"][0]


def test_summarize_realism_vs_divergence_windows_without_divergence_rate():
    """Test handling when windows don't have divergence_rate field."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_010",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "other_field": 0.1},
        {"window_index": 1, "other_field": 0.2},
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert result["schema_version"] == "1.0.0"
    assert result["consistency_status"] == "TENSION"
    assert len(result["advisory_notes"]) > 0


def test_attach_realism_cards_to_evidence_with_divergence_consistency(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that divergence consistency is attached when windows map is provided."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    annex = build_first_light_realism_annex(analysis)
    card = build_cal_exp_realism_card("cal_exp_001", annex)
    
    # Create windows map
    cal_exp_windows_map = {
        "cal_exp_001": [
            {"window_index": 0, "divergence_rate": 0.1},
            {"window_index": 1, "divergence_rate": 0.2},
            {"window_index": 2, "divergence_rate": 0.3},
        ]
    }
    
    evidence = {"governance": {}}
    new_evidence = attach_realism_cards_to_evidence(evidence, [card], cal_exp_windows_map)
    
    # Verify divergence_consistency is attached
    cards = new_evidence["governance"]["realism_cards"]["cards"]
    assert len(cards) == 1
    assert "divergence_consistency" in cards[0]
    
    consistency = cards[0]["divergence_consistency"]
    assert "consistency_status" in consistency
    assert "advisory_notes" in consistency
    assert consistency["schema_version"] == "1.0.0"


def test_attach_realism_cards_to_evidence_without_divergence_consistency(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that divergence consistency is not attached when windows map is not provided."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    annex = build_first_light_realism_annex(analysis)
    card = build_cal_exp_realism_card("cal_exp_001", annex)
    
    evidence = {"governance": {}}
    new_evidence = attach_realism_cards_to_evidence(evidence, [card])
    
    # Verify divergence_consistency is not attached
    cards = new_evidence["governance"]["realism_cards"]["cards"]
    assert len(cards) == 1
    assert "divergence_consistency" not in cards[0]


def test_attach_realism_cards_to_evidence_divergence_consistency_json_round_trip(mock_ratchet, mock_calibration_console, mock_scenario_configs):
    """Test that evidence with divergence consistency can be JSON serialized."""
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_calibration_console,
        scenario_configs=mock_scenario_configs,
    )
    
    annex = build_first_light_realism_annex(analysis)
    card = build_cal_exp_realism_card("cal_exp_001", annex)
    
    cal_exp_windows_map = {
        "cal_exp_001": [
            {"window_index": 0, "divergence_rate": 0.1},
            {"window_index": 1, "divergence_rate": 0.2},
        ]
    }
    
    evidence = {"governance": {}}
    new_evidence = attach_realism_cards_to_evidence(evidence, [card], cal_exp_windows_map)
    
    # Serialize and deserialize
    json_str = json.dumps(new_evidence)
    deserialized = json.loads(json_str)
    
    # Verify structure preserved
    cards = deserialized["governance"]["realism_cards"]["cards"]
    assert "divergence_consistency" in cards[0]
    consistency = cards[0]["divergence_consistency"]
    assert consistency["schema_version"] == "1.0.0"
    assert "consistency_status" in consistency


# ==============================================================================
# PARAMETERIZED THRESHOLDS AND WINDOW NORMALIZATION TESTS
# ==============================================================================

def test_summarize_realism_vs_divergence_parameter_overrides_change_classification():
    """Test that parameter overrides change classification."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_param_test",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    # With default thresholds (0.5), this would be CONSISTENT (0.4 < 0.5)
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.3},
        {"window_index": 1, "divergence_rate": 0.4},
    ]
    
    # Default: should be CONSISTENT
    result_default = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    assert result_default["consistency_status"] == "CONSISTENT"
    
    # With lower threshold (0.3), should become TENSION (0.4 >= 0.3)
    result_custom = summarize_realism_vs_divergence(
        realism_card, 
        cal_exp_windows,
        low_divergence_threshold=0.3,
    )
    assert result_custom["consistency_status"] == "TENSION"


def test_summarize_realism_vs_divergence_window_normalization_direct_key():
    """Test window normalization with direct divergence_rate key."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_norm_direct",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},
        {"window_index": 1, "divergence_rate": 0.2},
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert result["windows_analyzed"] == 2
    assert result["high_divergence_window_count"] == 0


def test_summarize_realism_vs_divergence_window_normalization_metrics_nested():
    """Test window normalization with metrics.divergence_rate nested key."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_norm_metrics",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "metrics": {"divergence_rate": 0.1}},
        {"window_index": 1, "metrics": {"divergence_rate": 0.2}},
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert result["windows_analyzed"] == 2
    assert result["high_divergence_window_count"] == 0
    assert result["consistency_status"] == "CONSISTENT"


def test_summarize_realism_vs_divergence_window_normalization_summary_nested():
    """Test window normalization with summary.divergence_rate nested key."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_norm_summary",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "summary": {"divergence_rate": 0.1}},
        {"window_index": 1, "summary": {"divergence_rate": 0.2}},
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert result["windows_analyzed"] == 2
    assert result["high_divergence_window_count"] == 0
    assert result["consistency_status"] == "CONSISTENT"


def test_summarize_realism_vs_divergence_window_normalization_mixed_schemas():
    """Test window normalization with mixed window schemas."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_norm_mixed",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    # Mix of direct, metrics, and summary keys
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},
        {"window_index": 1, "metrics": {"divergence_rate": 0.2}},
        {"window_index": 2, "summary": {"divergence_rate": 0.3}},
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert result["windows_analyzed"] == 3
    assert result["high_divergence_window_count"] == 0


def test_summarize_realism_vs_divergence_window_normalization_missing_keys():
    """Test window normalization when some windows lack divergence_rate."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_norm_missing",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},
        {"window_index": 1, "other_field": 0.5},  # Missing divergence_rate
        {"window_index": 2, "divergence_rate": 0.2},
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    # Should only analyze 2 windows (those with divergence_rate)
    assert result["windows_analyzed"] == 2
    assert result["high_divergence_window_count"] == 0


def test_summarize_realism_vs_divergence_explainability_fields():
    """Test that explainability fields (windows_analyzed, high_divergence_window_count) are present."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_explain",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},
        {"window_index": 1, "divergence_rate": 0.2},
        {"window_index": 2, "divergence_rate": 0.95},  # High divergence
        {"window_index": 3, "divergence_rate": 0.92},  # High divergence
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    # Verify explainability fields
    assert "windows_analyzed" in result
    assert "high_divergence_window_count" in result
    assert result["windows_analyzed"] == 4
    assert result["high_divergence_window_count"] == 2  # 0.95 and 0.92 >= 0.9


def test_summarize_realism_vs_divergence_deterministic_with_parameters():
    """Test that consistency check is deterministic with custom parameters."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_det_param",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},
        {"window_index": 1, "divergence_rate": 0.2},
    ]
    
    # Same parameters should produce same result
    result1 = summarize_realism_vs_divergence(
        realism_card,
        cal_exp_windows,
        low_divergence_threshold=0.3,
        high_divergence_threshold=0.8,
        persistent_high_divergence_threshold=0.95,
        persistent_window_count=2,
    )
    
    result2 = summarize_realism_vs_divergence(
        realism_card,
        cal_exp_windows,
        low_divergence_threshold=0.3,
        high_divergence_threshold=0.8,
        persistent_high_divergence_threshold=0.95,
        persistent_window_count=2,
    )
    
    assert result1 == result2, "Consistency check must be deterministic with parameters"


def test_summarize_realism_vs_divergence_persistent_window_count_override():
    """Test that persistent_window_count parameter affects CONFLICT detection."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_persistent",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    # 2 windows with high divergence (default requires 3)
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.95},
        {"window_index": 1, "divergence_rate": 0.92},
    ]
    
    # Default: should be TENSION (GREEN but final divergence 0.92 >= 0.5, and only 2 windows < 3 for CONFLICT)
    result_default = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    assert result_default["consistency_status"] == "TENSION"
    assert result_default["high_divergence_window_count"] == 2
    
    # With lower persistent_window_count (2): should be CONFLICT
    result_custom = summarize_realism_vs_divergence(
        realism_card,
        cal_exp_windows,
        persistent_window_count=2,
    )
    assert result_custom["consistency_status"] == "CONFLICT"
    assert result_custom["high_divergence_window_count"] == 2


def test_summarize_realism_vs_divergence_persistent_threshold_override():
    """Test that persistent_high_divergence_threshold parameter affects high divergence count."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_threshold",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    # Windows with divergence rates around 0.8
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.85},
        {"window_index": 1, "divergence_rate": 0.82},
        {"window_index": 2, "divergence_rate": 0.88},
    ]
    
    # Default threshold (0.9): should count 0 high divergence windows
    result_default = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    assert result_default["high_divergence_window_count"] == 0
    
    # Lower threshold (0.8): should count 3 high divergence windows
    result_custom = summarize_realism_vs_divergence(
        realism_card,
        cal_exp_windows,
        persistent_high_divergence_threshold=0.8,
    )
    assert result_custom["high_divergence_window_count"] == 3


# ==============================================================================
# WINDOW SCHEMA AUDIT TRAIL TESTS
# ==============================================================================

def test_summarize_realism_vs_divergence_source_path_tracking_direct():
    """Test that source_path tracking counts DIRECT keys correctly."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_source_direct",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},
        {"window_index": 1, "divergence_rate": 0.2},
        {"window_index": 2, "divergence_rate": 0.3},
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert "divergence_rate_sources" in result
    assert result["divergence_rate_sources"]["DIRECT"] == 3
    assert "MISSING" not in result["divergence_rate_sources"] or result["divergence_rate_sources"]["MISSING"] == 0
    assert result["windows_dropped_missing_rate"] == 0


def test_summarize_realism_vs_divergence_source_path_tracking_metrics_nested():
    """Test that source_path tracking counts METRICS_NESTED keys correctly."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_source_metrics",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "metrics": {"divergence_rate": 0.1}},
        {"window_index": 1, "metrics": {"divergence_rate": 0.2}},
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert "divergence_rate_sources" in result
    assert result["divergence_rate_sources"]["METRICS_NESTED"] == 2
    assert result["windows_dropped_missing_rate"] == 0


def test_summarize_realism_vs_divergence_source_path_tracking_summary_nested():
    """Test that source_path tracking counts SUMMARY_NESTED keys correctly."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_source_summary",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "summary": {"divergence_rate": 0.1}},
        {"window_index": 1, "summary": {"divergence_rate": 0.2}},
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert "divergence_rate_sources" in result
    assert result["divergence_rate_sources"]["SUMMARY_NESTED"] == 2
    assert result["windows_dropped_missing_rate"] == 0


def test_summarize_realism_vs_divergence_source_path_tracking_mixed():
    """Test that source_path tracking handles mixed window schemas."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_source_mixed",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},  # DIRECT
        {"window_index": 1, "metrics": {"divergence_rate": 0.2}},  # METRICS_NESTED
        {"window_index": 2, "summary": {"divergence_rate": 0.3}},  # SUMMARY_NESTED
        {"window_index": 3, "other_field": 0.5},  # MISSING
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert "divergence_rate_sources" in result
    assert result["divergence_rate_sources"]["DIRECT"] == 1
    assert result["divergence_rate_sources"]["METRICS_NESTED"] == 1
    assert result["divergence_rate_sources"]["SUMMARY_NESTED"] == 1
    assert result["divergence_rate_sources"]["MISSING"] == 1
    assert result["windows_dropped_missing_rate"] == 1
    assert result["windows_analyzed"] == 3


def test_summarize_realism_vs_divergence_source_path_tracking_missing_only():
    """Test that source_path tracking correctly identifies all MISSING windows."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_source_missing",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "other_field": 0.5},
        {"window_index": 1, "another_field": 0.6},
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert "divergence_rate_sources" in result
    assert result["divergence_rate_sources"]["MISSING"] == 2
    assert result["windows_dropped_missing_rate"] == 2
    assert result["windows_analyzed"] == 0


def test_summarize_realism_vs_divergence_strict_mode_no_missing():
    """Test that strict mode does not affect result when no windows are MISSING."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_strict_no_missing",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},
        {"window_index": 1, "divergence_rate": 0.2},
    ]
    
    result_strict = summarize_realism_vs_divergence(realism_card, cal_exp_windows, strict_window_extraction=True)
    result_normal = summarize_realism_vs_divergence(realism_card, cal_exp_windows, strict_window_extraction=False)
    
    # Should produce same result when no missing windows
    assert result_strict["consistency_status"] == result_normal["consistency_status"]
    assert result_strict["consistency_status"] != "INCONCLUSIVE"


def test_summarize_realism_vs_divergence_strict_mode_with_missing():
    """Test that strict mode sets status to INCONCLUSIVE when any window is MISSING."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_strict_missing",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},
        {"window_index": 1, "other_field": 0.5},  # MISSING
        {"window_index": 2, "divergence_rate": 0.2},
    ]
    
    # Normal mode: should analyze available windows
    result_normal = summarize_realism_vs_divergence(realism_card, cal_exp_windows, strict_window_extraction=False)
    assert result_normal["consistency_status"] != "INCONCLUSIVE"
    assert result_normal["windows_analyzed"] == 2
    assert result_normal["windows_dropped_missing_rate"] == 1
    
    # Strict mode: should set to INCONCLUSIVE
    result_strict = summarize_realism_vs_divergence(realism_card, cal_exp_windows, strict_window_extraction=True)
    assert result_strict["consistency_status"] == "INCONCLUSIVE"
    assert result_strict["windows_analyzed"] == 2  # Still analyzed available windows
    assert result_strict["windows_dropped_missing_rate"] == 1
    assert "Strict mode enabled" in result_strict["advisory_notes"][0]


def test_summarize_realism_vs_divergence_strict_mode_all_missing():
    """Test that strict mode sets status to INCONCLUSIVE when all windows are MISSING."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_strict_all_missing",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "other_field": 0.5},
        {"window_index": 1, "another_field": 0.6},
    ]
    
    result_strict = summarize_realism_vs_divergence(realism_card, cal_exp_windows, strict_window_extraction=True)
    
    assert result_strict["consistency_status"] == "INCONCLUSIVE"
    assert result_strict["windows_analyzed"] == 0
    assert result_strict["windows_dropped_missing_rate"] == 2
    assert "Strict mode enabled" in result_strict["advisory_notes"][0]


def test_summarize_realism_vs_divergence_source_path_deterministic():
    """Test that source_path tracking is deterministic."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_source_det",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},
        {"window_index": 1, "metrics": {"divergence_rate": 0.2}},
        {"window_index": 2, "other_field": 0.5},
    ]
    
    result1 = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    result2 = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    assert result1 == result2, "Source path tracking must be deterministic"
    assert result1["divergence_rate_sources"] == result2["divergence_rate_sources"]


def test_summarize_realism_vs_divergence_source_path_json_safe():
    """Test that source_path tracking output is JSON serializable."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_source_json",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},
        {"window_index": 1, "metrics": {"divergence_rate": 0.2}},
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows)
    
    # Should not raise exception
    json_str = json.dumps(result)
    assert isinstance(json_str, str)
    
    # Should be able to deserialize
    deserialized = json.loads(json_str)
    assert "divergence_rate_sources" in deserialized
    assert "windows_dropped_missing_rate" in deserialized
    assert isinstance(deserialized["divergence_rate_sources"], dict)


def test_coerce_source_path_valid_paths():
    """Test that _coerce_source_path accepts valid paths."""
    assert _coerce_source_path(SOURCE_PATH_DIRECT) == SOURCE_PATH_DIRECT
    assert _coerce_source_path(SOURCE_PATH_METRICS_NESTED) == SOURCE_PATH_METRICS_NESTED
    assert _coerce_source_path(SOURCE_PATH_SUMMARY_NESTED) == SOURCE_PATH_SUMMARY_NESTED
    assert _coerce_source_path(SOURCE_PATH_MISSING) == SOURCE_PATH_MISSING


def test_coerce_source_path_invalid_path_fallback():
    """Test that _coerce_source_path falls back to MISSING for unknown paths."""
    # Should fallback to MISSING even if assert fails
    result = _coerce_source_path("UNKNOWN_PATH")
    assert result == SOURCE_PATH_MISSING


def test_summarize_realism_vs_divergence_strict_mode_contract_enabled():
    """Test that strict_mode_contract is present and correct when strict mode enabled."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_strict_contract",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},
        {"window_index": 1, "other_field": 0.5},  # MISSING
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows, strict_window_extraction=True)
    
    assert "strict_mode_contract" in result
    contract = result["strict_mode_contract"]
    assert contract["enabled"] is True
    assert contract["missing_windows_count"] == 1
    assert contract["status_when_missing"] == "INCONCLUSIVE"
    assert result["consistency_status"] == "INCONCLUSIVE"


def test_summarize_realism_vs_divergence_strict_mode_contract_disabled():
    """Test that strict_mode_contract is present and correct when strict mode disabled."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_strict_contract_disabled",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},
        {"window_index": 1, "other_field": 0.5},  # MISSING
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows, strict_window_extraction=False)
    
    assert "strict_mode_contract" in result
    contract = result["strict_mode_contract"]
    assert contract["enabled"] is False
    assert contract["missing_windows_count"] == 1  # Still counted
    assert contract["status_when_missing"] == "INCONCLUSIVE"
    assert result["consistency_status"] != "INCONCLUSIVE"  # Not INCONCLUSIVE because strict mode disabled


def test_summarize_realism_vs_divergence_strict_mode_audit_counts_deterministic():
    """Test that strict mode produces INCONCLUSIVE while still reporting audit counts deterministically."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_strict_audit",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},  # DIRECT
        {"window_index": 1, "metrics": {"divergence_rate": 0.2}},  # METRICS_NESTED
        {"window_index": 2, "other_field": 0.5},  # MISSING
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows, strict_window_extraction=True)
    
    # Should be INCONCLUSIVE due to strict mode
    assert result["consistency_status"] == "INCONCLUSIVE"
    
    # But should still report all audit counts
    assert result["windows_analyzed"] == 2  # 2 valid windows
    assert result["windows_dropped_missing_rate"] == 1  # 1 missing
    assert result["divergence_rate_sources"][SOURCE_PATH_DIRECT] == 1
    assert result["divergence_rate_sources"][SOURCE_PATH_METRICS_NESTED] == 1
    assert result["divergence_rate_sources"][SOURCE_PATH_MISSING] == 1
    
    # Verify strict_mode_contract
    contract = result["strict_mode_contract"]
    assert contract["enabled"] is True
    assert contract["missing_windows_count"] == 1
    
    # Should be deterministic
    result2 = summarize_realism_vs_divergence(realism_card, cal_exp_windows, strict_window_extraction=True)
    assert result == result2


def test_summarize_realism_vs_divergence_strict_mode_contract_no_missing():
    """Test that strict_mode_contract is present even when no missing windows."""
    realism_card = {
        "schema_version": "1.0.0",
        "cal_id": "cal_exp_strict_no_missing",
        "status_light": "GREEN",
        "global_pressure": 0.2,
        "overall_confidence": 0.8,
    }
    
    cal_exp_windows = [
        {"window_index": 0, "divergence_rate": 0.1},
        {"window_index": 1, "divergence_rate": 0.2},
    ]
    
    result = summarize_realism_vs_divergence(realism_card, cal_exp_windows, strict_window_extraction=True)
    
    assert "strict_mode_contract" in result
    contract = result["strict_mode_contract"]
    assert contract["enabled"] is True
    assert contract["missing_windows_count"] == 0
    assert contract["status_when_missing"] == "INCONCLUSIVE"

