"""
Tests for Phase VI: Readiness Tensor Engine v1

This module tests:
  - Tensor math and normalization
  - Norm ordering and ranking
  - Polygraph logic
  - Sentinels and director tile determinism
  - Neutral language checks
  - No uplift semantics
"""

import json
import math
import unittest
from typing import Dict, Any

from experiments.metric_adapter_introspection import (
    build_metric_readiness_tensor,
    build_metric_drift_polygraph,
    build_metric_readiness_autopilot_director_panel,
    evaluate_phase_transition_safety_v2,
    READINESS_TENSOR_SCHEMA_VERSION,
    TransitionBand,
    StatusLight,
)


# =============================================================================
# Test Fixtures
# =============================================================================

def create_sample_readiness_matrix() -> Dict[str, Any]:
    """Create a sample readiness matrix for testing."""
    return {
        "schema_version": "1.0.0",
        "matrix": {
            "slice_a": {
                "goal_hit": {
                    "status": "READY",
                    "drift_severity": "NONE",
                    "ready_for_experiments": True
                }
            },
            "slice_b": {
                "sparse_success": {
                    "status": "DEGRADED",
                    "drift_severity": "PARAMETRIC_MINOR",
                    "ready_for_experiments": False
                }
            },
            "slice_c": {
                "chain_success": {
                    "status": "BLOCKED",
                    "drift_severity": "SEMANTIC",
                    "ready_for_experiments": False
                }
            }
        }
    }


def create_sample_drift_grid() -> Dict[str, Any]:
    """Create a sample drift grid for testing."""
    return {
        "grid": {
            "slice_a": {
                "goal_hit": {"drift_status": "OK"}
            },
            "slice_b": {
                "sparse_success": {"drift_status": "WARN"}
            },
            "slice_c": {
                "chain_success": {"drift_status": "DRIFTY"}
            }
        }
    }


def create_sample_budget_view() -> Dict[str, Any]:
    """Create a sample budget joint view for testing."""
    return {
        "view": {
            "slice_a": {
                "goal_hit": {"budget_flag": "SAFE"}
            },
            "slice_b": {
                "sparse_success": {"budget_flag": "TIGHT"}
            },
            "slice_c": {
                "chain_success": {"budget_flag": "STARVED"}
            }
        }
    }


# =============================================================================
# Task 1: Tensor Math Tests
# =============================================================================

class TestBuildMetricReadinessTensor(unittest.TestCase):
    """Tests for build_metric_readiness_tensor function."""
    
    def test_tensor_has_schema_version(self):
        """Tensor output includes schema_version."""
        matrix = create_sample_readiness_matrix()
        drift = create_sample_drift_grid()
        budget = create_sample_budget_view()
        
        tensor = build_metric_readiness_tensor(matrix, drift, budget)
        
        self.assertIn("schema_version", tensor)
        self.assertEqual(tensor["schema_version"], READINESS_TENSOR_SCHEMA_VERSION)
    
    def test_tensor_has_required_keys(self):
        """Tensor has slice_vectors, global_norm, ranked_slices."""
        matrix = create_sample_readiness_matrix()
        drift = create_sample_drift_grid()
        budget = create_sample_budget_view()
        
        tensor = build_metric_readiness_tensor(matrix, drift, budget)
        
        self.assertIn("slice_vectors", tensor)
        self.assertIn("global_norm", tensor)
        self.assertIn("ranked_slices", tensor)
    
    def test_slice_vectors_have_required_fields(self):
        """Each slice vector has required components."""
        matrix = create_sample_readiness_matrix()
        drift = create_sample_drift_grid()
        budget = create_sample_budget_view()
        
        tensor = build_metric_readiness_tensor(matrix, drift, budget)
        
        for slice_name, vector in tensor["slice_vectors"].items():
            self.assertIn("readiness_score", vector)
            self.assertIn("drift_component", vector)
            self.assertIn("budget_component", vector)
            self.assertIn("metric_consistency_component", vector)
            self.assertIn("vector_norm", vector)
    
    def test_components_normalized_to_0_1(self):
        """All components are normalized to [0, 1] range."""
        matrix = create_sample_readiness_matrix()
        drift = create_sample_drift_grid()
        budget = create_sample_budget_view()
        
        tensor = build_metric_readiness_tensor(matrix, drift, budget)
        
        for slice_name, vector in tensor["slice_vectors"].items():
            self.assertGreaterEqual(vector["readiness_score"], 0.0)
            self.assertLessEqual(vector["readiness_score"], 1.0)
            self.assertGreaterEqual(vector["drift_component"], 0.0)
            self.assertLessEqual(vector["drift_component"], 1.0)
            self.assertGreaterEqual(vector["budget_component"], 0.0)
            self.assertLessEqual(vector["budget_component"], 1.0)
            self.assertGreaterEqual(vector["metric_consistency_component"], 0.0)
            self.assertLessEqual(vector["metric_consistency_component"], 1.0)
    
    def test_readiness_score_weighted_correctly(self):
        """readiness_score uses correct weights (0.5 readiness, 0.25 drift, 0.25 budget)."""
        matrix = {
            "schema_version": "1.0.0",
            "matrix": {
                "slice_test": {
                    "goal_hit": {"status": "READY", "drift_severity": "NONE", "ready_for_experiments": True}
                }
            }
        }
        drift = {
            "grid": {
                "slice_test": {
                    "goal_hit": {"drift_status": "OK"}
                }
            }
        }
        budget = {
            "view": {
                "slice_test": {
                    "goal_hit": {"budget_flag": "SAFE"}
                }
            }
        }
        
        tensor = build_metric_readiness_tensor(matrix, drift, budget)
        
        vector = tensor["slice_vectors"]["slice_test"]
        # READY=1.0, OK=1.0, SAFE=1.0
        # Score = 0.5*1.0 + 0.25*1.0 + 0.25*1.0 = 1.0
        self.assertAlmostEqual(vector["readiness_score"], 1.0, places=2)
    
    def test_vector_norm_computed(self):
        """vector_norm is computed as L2 norm."""
        matrix = {
            "schema_version": "1.0.0",
            "matrix": {
                "slice_test": {
                    "goal_hit": {"status": "READY", "drift_severity": "NONE", "ready_for_experiments": True}
                }
            }
        }
        drift = {
            "grid": {
                "slice_test": {
                    "goal_hit": {"drift_status": "OK"}
                }
            }
        }
        budget = {
            "view": {
                "slice_test": {
                    "goal_hit": {"budget_flag": "SAFE"}
                }
            }
        }
        
        tensor = build_metric_readiness_tensor(matrix, drift, budget)
        
        vector = tensor["slice_vectors"]["slice_test"]
        # All components = 1.0, consistency = 1.0
        # Norm = sqrt(1^2 + 1^2 + 1^2 + 1^2) = sqrt(4) = 2.0
        expected_norm = math.sqrt(4.0)
        self.assertAlmostEqual(vector["vector_norm"], expected_norm, places=2)
    
    def test_global_norm_computed(self):
        """global_norm is average of slice norms."""
        matrix = create_sample_readiness_matrix()
        drift = create_sample_drift_grid()
        budget = create_sample_budget_view()
        
        tensor = build_metric_readiness_tensor(matrix, drift, budget)
        
        slice_norms = [v["vector_norm"] for v in tensor["slice_vectors"].values()]
        expected_global = sum(slice_norms) / len(slice_norms) if slice_norms else 0.0
        
        self.assertAlmostEqual(tensor["global_norm"], expected_global, places=2)
    
    def test_ranked_slices_descending(self):
        """ranked_slices are sorted by vector_norm descending."""
        matrix = create_sample_readiness_matrix()
        drift = create_sample_drift_grid()
        budget = create_sample_budget_view()
        
        tensor = build_metric_readiness_tensor(matrix, drift, budget)
        
        ranked = tensor["ranked_slices"]
        norms = [tensor["slice_vectors"][s]["vector_norm"] for s in ranked]
        
        # Check descending order
        for i in range(len(norms) - 1):
            self.assertGreaterEqual(norms[i], norms[i + 1])
    
    def test_deterministic_output(self):
        """Tensor output is deterministic."""
        matrix = create_sample_readiness_matrix()
        drift = create_sample_drift_grid()
        budget = create_sample_budget_view()
        
        tensors = [build_metric_readiness_tensor(matrix, drift, budget) for _ in range(5)]
        json_outputs = [json.dumps(t, sort_keys=True) for t in tensors]
        
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))
    
    def test_handles_missing_drift_budget(self):
        """Tensor handles missing drift/budget data gracefully."""
        matrix = create_sample_readiness_matrix()
        drift = {"grid": {}}  # Missing data
        budget = {"view": {}}  # Missing data
        
        tensor = build_metric_readiness_tensor(matrix, drift, budget)
        
        # Should still produce valid tensor with defaults
        self.assertIn("slice_vectors", tensor)
        self.assertGreater(len(tensor["slice_vectors"]), 0)


# =============================================================================
# Task 2: Polygraph Logic Tests
# =============================================================================

class TestBuildMetricDriftPolygraph(unittest.TestCase):
    """Tests for build_metric_drift_polygraph function."""
    
    def test_polygraph_has_required_fields(self):
        """Polygraph has all required fields."""
        tensor = {
            "slice_vectors": {
                "slice_a": {
                    "readiness_score": 0.8,
                    "drift_component": 0.9,
                    "budget_component": 0.7,
                    "metric_consistency_component": 1.0,
                    "vector_norm": 1.5
                }
            },
            "global_norm": 1.5
        }
        history = {"windows": []}
        
        polygraph = build_metric_drift_polygraph(tensor, history)
        
        self.assertIn("drift_momentum", polygraph)
        self.assertIn("entangled_pairs", polygraph)
        self.assertIn("poly_fail_detected", polygraph)
        self.assertIn("neutral_notes", polygraph)
    
    def test_drift_momentum_computed_from_history(self):
        """drift_momentum computed from history windows."""
        tensor = {
            "slice_vectors": {
                "slice_a": {
                    "readiness_score": 0.5,
                    "drift_component": 0.3,  # Degraded
                    "budget_component": 0.5,
                    "metric_consistency_component": 1.0,
                    "vector_norm": 1.0
                }
            },
            "global_norm": 1.0
        }
        history = {
            "windows": [
                {
                    "timestamp": "2024-01-01",
                    "tensor": {
                        "slice_vectors": {
                            "slice_a": {
                                "drift_component": 0.8  # Was better
                            }
                        }
                    }
                },
                {
                    "timestamp": "2024-01-02",
                    "tensor": {
                        "slice_vectors": {
                            "slice_a": {
                                "drift_component": 0.3  # Now worse
                            }
                        }
                    }
                }
            ]
        }
        
        polygraph = build_metric_drift_polygraph(tensor, history)
        
        # Momentum should be negative (degrading)
        self.assertLess(polygraph["drift_momentum"], 0.0)
    
    def test_poly_fail_detected_when_all_low(self):
        """poly_fail_detected when all components below threshold."""
        tensor = {
            "slice_vectors": {
                "slice_a": {
                    "readiness_score": 0.3,  # < 0.4
                    "drift_component": 0.3,  # < 0.4
                    "budget_component": 0.3,  # < 0.4
                    "metric_consistency_component": 1.0,
                    "vector_norm": 1.0
                }
            },
            "global_norm": 1.0
        }
        history = {"windows": []}
        
        polygraph = build_metric_drift_polygraph(tensor, history)
        
        self.assertTrue(polygraph["poly_fail_detected"])
        self.assertIn("slice_a", polygraph["neutral_notes"][0])
    
    def test_entangled_pairs_detected(self):
        """entangled_pairs detects slices with similar drift patterns."""
        tensor = {
            "slice_vectors": {
                "slice_a": {
                    "readiness_score": 0.5,
                    "drift_component": 0.3,  # Similar
                    "budget_component": 0.5,
                    "metric_consistency_component": 1.0,
                    "vector_norm": 1.0
                },
                "slice_b": {
                    "readiness_score": 0.5,
                    "drift_component": 0.35,  # Similar (within 0.1)
                    "budget_component": 0.5,
                    "metric_consistency_component": 1.0,
                    "vector_norm": 1.0
                }
            },
            "global_norm": 1.0
        }
        history = {"windows": []}
        
        polygraph = build_metric_drift_polygraph(tensor, history)
        
        # Should detect entanglement
        self.assertGreater(len(polygraph["entangled_pairs"]), 0)
    
    def test_neutral_notes_included(self):
        """neutral_notes includes explanations."""
        tensor = {
            "slice_vectors": {
                "slice_a": {
                    "readiness_score": 0.3,
                    "drift_component": 0.3,
                    "budget_component": 0.3,
                    "metric_consistency_component": 1.0,
                    "vector_norm": 1.0
                }
            },
            "global_norm": 1.0
        }
        history = {"windows": []}
        
        polygraph = build_metric_drift_polygraph(tensor, history)
        
        self.assertGreater(len(polygraph["neutral_notes"]), 0)
    
    def test_deterministic_output(self):
        """Polygraph output is deterministic."""
        tensor = {
            "slice_vectors": {
                "slice_a": {
                    "readiness_score": 0.5,
                    "drift_component": 0.5,
                    "budget_component": 0.5,
                    "metric_consistency_component": 1.0,
                    "vector_norm": 1.0
                }
            },
            "global_norm": 1.0
        }
        history = {"windows": []}
        
        polygraphs = [build_metric_drift_polygraph(tensor, history) for _ in range(5)]
        json_outputs = [json.dumps(p, sort_keys=True) for p in polygraphs]
        
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))


# =============================================================================
# Task 3: Director Tile v2 Tests
# =============================================================================

class TestBuildMetricReadinessAutopilotDirectorPanel(unittest.TestCase):
    """Tests for build_metric_readiness_autopilot_director_panel function."""
    
    def test_panel_has_required_fields(self):
        """Panel has all required fields."""
        tensor = {
            "global_norm": 0.8,
            "slice_vectors": {}
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": 0.0
        }
        autopilot = {
            "autopilot_status": "OK",
            "slices_to_hold": [],
            "slices_safe_to_progress": ["slice_a.goal_hit"]
        }
        
        panel = build_metric_readiness_autopilot_director_panel(tensor, polygraph, autopilot)
        
        self.assertIn("status_light", panel)
        self.assertIn("autopilot_status", panel)
        self.assertIn("global_norm", panel)
        self.assertIn("poly_fail_detected", panel)
        self.assertIn("slices_to_hold", panel)
        self.assertIn("slices_safe_to_progress", panel)
        self.assertIn("headline", panel)
    
    def test_status_light_red_when_poly_fail(self):
        """status_light is RED when poly_fail_detected."""
        tensor = {
            "global_norm": 0.8,
            "slice_vectors": {}
        }
        polygraph = {
            "poly_fail_detected": True,
            "drift_momentum": 0.0
        }
        autopilot = {
            "autopilot_status": "OK",
            "slices_to_hold": [],
            "slices_safe_to_progress": []
        }
        
        panel = build_metric_readiness_autopilot_director_panel(tensor, polygraph, autopilot)
        
        self.assertEqual(panel["status_light"], StatusLight.RED.value)
    
    def test_status_light_red_when_low_norm(self):
        """status_light is RED when global_norm < 0.35."""
        tensor = {
            "global_norm": 0.3,  # < 0.35
            "slice_vectors": {}
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": 0.0
        }
        autopilot = {
            "autopilot_status": "OK",
            "slices_to_hold": [],
            "slices_safe_to_progress": []
        }
        
        panel = build_metric_readiness_autopilot_director_panel(tensor, polygraph, autopilot)
        
        self.assertEqual(panel["status_light"], StatusLight.RED.value)
    
    def test_status_light_yellow_when_attention(self):
        """status_light is YELLOW when autopilot is ATTENTION."""
        tensor = {
            "global_norm": 0.8,
            "slice_vectors": {}
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": 0.0
        }
        autopilot = {
            "autopilot_status": "ATTENTION",
            "slices_to_hold": [],
            "slices_safe_to_progress": []
        }
        
        panel = build_metric_readiness_autopilot_director_panel(tensor, polygraph, autopilot)
        
        self.assertEqual(panel["status_light"], StatusLight.YELLOW.value)
    
    def test_status_light_yellow_when_negative_momentum(self):
        """status_light is YELLOW when drift_momentum < -0.1."""
        tensor = {
            "global_norm": 0.8,
            "slice_vectors": {}
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": -0.15  # < -0.1
        }
        autopilot = {
            "autopilot_status": "OK",
            "slices_to_hold": [],
            "slices_safe_to_progress": []
        }
        
        panel = build_metric_readiness_autopilot_director_panel(tensor, polygraph, autopilot)
        
        self.assertEqual(panel["status_light"], StatusLight.YELLOW.value)
    
    def test_status_light_green_when_stable(self):
        """status_light is GREEN when all stable."""
        tensor = {
            "global_norm": 0.8,
            "slice_vectors": {}
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": 0.0
        }
        autopilot = {
            "autopilot_status": "OK",
            "slices_to_hold": [],
            "slices_safe_to_progress": ["slice_a.goal_hit"]
        }
        
        panel = build_metric_readiness_autopilot_director_panel(tensor, polygraph, autopilot)
        
        self.assertEqual(panel["status_light"], StatusLight.GREEN.value)
    
    def test_headline_neutral_language(self):
        """headline uses neutral language."""
        tensor = {
            "global_norm": 0.8,
            "slice_vectors": {}
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": 0.0
        }
        autopilot = {
            "autopilot_status": "OK",
            "slices_to_hold": [],
            "slices_safe_to_progress": []
        }
        
        panel = build_metric_readiness_autopilot_director_panel(tensor, polygraph, autopilot)
        
        headline = panel["headline"]
        # Check for absence of value judgment words
        self.assertNotIn("good", headline.lower())
        self.assertNotIn("bad", headline.lower())
        self.assertNotIn("better", headline.lower())
        self.assertNotIn("worse", headline.lower())
    
    def test_deterministic_output(self):
        """Panel output is deterministic."""
        tensor = {
            "global_norm": 0.8,
            "slice_vectors": {}
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": 0.0
        }
        autopilot = {
            "autopilot_status": "OK",
            "slices_to_hold": [],
            "slices_safe_to_progress": []
        }
        
        panels = [
            build_metric_readiness_autopilot_director_panel(tensor, polygraph, autopilot)
            for _ in range(5)
        ]
        json_outputs = [json.dumps(p, sort_keys=True) for p in panels]
        
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))


# =============================================================================
# Task 4: Phase Transition Sentinel Tests
# =============================================================================

class TestEvaluatePhaseTransitionSafetyV2(unittest.TestCase):
    """Tests for evaluate_phase_transition_safety_v2 function."""
    
    def test_sentinel_has_required_fields(self):
        """Sentinel has all required fields."""
        tensor = {
            "global_norm": 0.8
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": 0.0
        }
        promotions = {
            "promotion_ok": True
        }
        
        sentinel = evaluate_phase_transition_safety_v2(tensor, polygraph, promotions)
        
        self.assertIn("transition_safe", sentinel)
        self.assertIn("blocking_conditions", sentinel)
        self.assertIn("transition_band", sentinel)
        self.assertIn("recommendations", sentinel)
    
    def test_transition_safe_when_all_conditions_met(self):
        """transition_safe is True when all conditions met."""
        tensor = {
            "global_norm": 0.8  # >= 0.35
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": 0.0  # Not strongly negative
        }
        promotions = {
            "promotion_ok": True
        }
        
        sentinel = evaluate_phase_transition_safety_v2(tensor, polygraph, promotions)
        
        self.assertTrue(sentinel["transition_safe"])
    
    def test_transition_not_safe_when_poly_fail(self):
        """transition_safe is False when poly_fail_detected."""
        tensor = {
            "global_norm": 0.8
        }
        polygraph = {
            "poly_fail_detected": True,
            "drift_momentum": 0.0
        }
        promotions = {
            "promotion_ok": True
        }
        
        sentinel = evaluate_phase_transition_safety_v2(tensor, polygraph, promotions)
        
        self.assertFalse(sentinel["transition_safe"])
        self.assertIn("Poly-fail", sentinel["blocking_conditions"][0])
    
    def test_transition_not_safe_when_low_norm(self):
        """transition_safe is False when global_norm < 0.35."""
        tensor = {
            "global_norm": 0.3  # < 0.35
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": 0.0
        }
        promotions = {
            "promotion_ok": True
        }
        
        sentinel = evaluate_phase_transition_safety_v2(tensor, polygraph, promotions)
        
        self.assertFalse(sentinel["transition_safe"])
        self.assertTrue(any("norm" in cond.lower() for cond in sentinel["blocking_conditions"]))
    
    def test_transition_not_safe_when_strong_momentum(self):
        """transition_safe is False when drift_momentum < -0.2."""
        tensor = {
            "global_norm": 0.8
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": -0.25  # < -0.2
        }
        promotions = {
            "promotion_ok": True
        }
        
        sentinel = evaluate_phase_transition_safety_v2(tensor, polygraph, promotions)
        
        self.assertFalse(sentinel["transition_safe"])
        self.assertTrue(any("momentum" in cond.lower() for cond in sentinel["blocking_conditions"]))
    
    def test_transition_band_high_when_conditions_met(self):
        """transition_band is HIGH when norm >= 0.7 and no poly_fail."""
        tensor = {
            "global_norm": 0.75  # >= 0.7
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": 0.0
        }
        promotions = {
            "promotion_ok": True
        }
        
        sentinel = evaluate_phase_transition_safety_v2(tensor, polygraph, promotions)
        
        self.assertEqual(sentinel["transition_band"], TransitionBand.HIGH.value)
    
    def test_transition_band_medium_when_moderate(self):
        """transition_band is MEDIUM when norm >= 0.5 and < 0.7."""
        tensor = {
            "global_norm": 0.6  # >= 0.5, < 0.7
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": 0.0
        }
        promotions = {
            "promotion_ok": True
        }
        
        sentinel = evaluate_phase_transition_safety_v2(tensor, polygraph, promotions)
        
        self.assertEqual(sentinel["transition_band"], TransitionBand.MEDIUM.value)
    
    def test_transition_band_low_when_poor(self):
        """transition_band is LOW when norm < 0.5 or poly_fail."""
        tensor = {
            "global_norm": 0.4  # < 0.5
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": 0.0
        }
        promotions = {
            "promotion_ok": True
        }
        
        sentinel = evaluate_phase_transition_safety_v2(tensor, polygraph, promotions)
        
        self.assertEqual(sentinel["transition_band"], TransitionBand.LOW.value)
    
    def test_recommendations_included(self):
        """recommendations includes appropriate actions."""
        tensor = {
            "global_norm": 0.3
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": 0.0
        }
        promotions = {
            "promotion_ok": True
        }
        
        sentinel = evaluate_phase_transition_safety_v2(tensor, polygraph, promotions)
        
        self.assertGreater(len(sentinel["recommendations"]), 0)
    
    def test_recommendations_neutral_language(self):
        """recommendations use neutral language."""
        tensor = {
            "global_norm": 0.8
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": 0.0
        }
        promotions = {
            "promotion_ok": True
        }
        
        sentinel = evaluate_phase_transition_safety_v2(tensor, polygraph, promotions)
        
        for rec in sentinel["recommendations"]:
            self.assertNotIn("good", rec.lower())
            self.assertNotIn("bad", rec.lower())
            self.assertNotIn("better", rec.lower())
            self.assertNotIn("worse", rec.lower())
    
    def test_deterministic_output(self):
        """Sentinel output is deterministic."""
        tensor = {
            "global_norm": 0.8
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": 0.0
        }
        promotions = {
            "promotion_ok": True
        }
        
        sentinels = [
            evaluate_phase_transition_safety_v2(tensor, polygraph, promotions)
            for _ in range(5)
        ]
        json_outputs = [json.dumps(s, sort_keys=True) for s in sentinels]
        
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))
    
    def test_no_uplift_semantics(self):
        """Output contains no uplift semantics."""
        tensor = {
            "global_norm": 0.8
        }
        polygraph = {
            "poly_fail_detected": False,
            "drift_momentum": 0.0
        }
        promotions = {
            "promotion_ok": True
        }
        
        sentinel = evaluate_phase_transition_safety_v2(tensor, polygraph, promotions)
        
        # Check that no uplift-related terms appear
        sentinel_str = json.dumps(sentinel).lower()
        self.assertNotIn("uplift", sentinel_str)
        self.assertNotIn("delta", sentinel_str)
        self.assertNotIn("p-value", sentinel_str)
        self.assertNotIn("significance", sentinel_str)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

