#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Test Suite for Scenario Causal Graph
--------------------------------------

Tests for:
    - Deterministic ordering
    - Purely structural (no effect on sampling)
    - Cycle detection
    - Graph visualization

NOT derived from real derivations; NOT part of Evidence Pack.

==============================================================================
"""

import json
import pytest
import sys
import tempfile
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL
from experiments.synthetic_uplift.causal_graph import (
    CausalLink,
    ScenarioCausalGraph,
    visualize_scenario_graph,
    build_default_causal_graph,
    load_causal_graph_from_registry,
    save_causal_graph_to_file,
)


# ==============================================================================
# DETERMINISTIC ORDERING TESTS
# ==============================================================================

class TestDeterministicOrdering:
    """Tests for deterministic topological ordering."""
    
    def test_topological_sort_deterministic(self):
        """Topological sort should produce same order every time."""
        graph = ScenarioCausalGraph()
        graph.add_scenario("synthetic_a")
        graph.add_scenario("synthetic_b")
        graph.add_scenario("synthetic_c")
        graph.add_link("synthetic_a", "synthetic_b")
        graph.add_link("synthetic_b", "synthetic_c")
        
        order1 = graph.topological_sort()
        order2 = graph.topological_sort()
        
        assert order1 == order2
    
    def test_topological_sort_respects_dependencies(self):
        """Dependencies should come before dependents."""
        graph = ScenarioCausalGraph()
        graph.add_scenario("synthetic_base")
        graph.add_scenario("synthetic_derived")
        graph.add_link("synthetic_base", "synthetic_derived")
        
        order = graph.topological_sort()
        
        assert order.index("synthetic_base") < order.index("synthetic_derived")
    
    def test_multiple_runs_same_order(self):
        """Multiple topological sorts should produce identical results."""
        graph = build_default_causal_graph()
        
        orders = [graph.topological_sort() for _ in range(5)]
        
        for order in orders[1:]:
            assert order == orders[0]


# ==============================================================================
# STRUCTURAL TESTS
# ==============================================================================

class TestPurelyStructural:
    """Tests ensuring graph is purely structural."""
    
    def test_graph_does_not_affect_outcomes(self):
        """Graph should only affect ordering, not sampling outcomes."""
        graph = ScenarioCausalGraph()
        graph.add_scenario("synthetic_test")
        
        # Graph should have no outcome-related attributes
        assert not hasattr(graph, "success_rate")
        assert not hasattr(graph, "probability")
        assert not hasattr(graph, "outcome")
    
    def test_links_are_ordering_only(self):
        """Links should only describe ordering constraints."""
        link = CausalLink(
            source="synthetic_a",
            target="synthetic_b",
            description="ordering only",
        )
        
        data = link.to_dict()
        
        # Should not contain outcome-related keys
        assert "probability" not in data
        assert "success" not in data
        assert "uplift" not in data
    
    def test_graph_serialization_structural(self):
        """Graph serialization should be purely structural."""
        graph = build_default_causal_graph()
        data = graph.to_dict()
        
        assert data["label"] == SAFETY_LABEL
        assert "scenarios" in data
        assert "links" in data
        
        # Should not contain empirical claims
        # Note: scenario names may contain "uplift" but descriptions should not claim outcomes
        for link in data.get("links", []):
            desc = (link.get("description") or "").lower()
            assert "positive" not in desc or "uplift" not in desc
            assert "negative" not in desc or "uplift" not in desc
            assert "leads to" not in desc


# ==============================================================================
# CYCLE DETECTION TESTS
# ==============================================================================

class TestCycleDetection:
    """Tests for cycle detection."""
    
    def test_no_cycle_acyclic_graph(self):
        """Acyclic graph should not have cycles."""
        graph = ScenarioCausalGraph()
        graph.add_scenario("synthetic_a")
        graph.add_scenario("synthetic_b")
        graph.add_scenario("synthetic_c")
        graph.add_link("synthetic_a", "synthetic_b")
        graph.add_link("synthetic_b", "synthetic_c")
        
        assert not graph.has_cycle()
    
    def test_detects_direct_cycle(self):
        """Should detect direct cycles (A -> B -> A)."""
        graph = ScenarioCausalGraph()
        graph.add_scenario("synthetic_a")
        graph.add_scenario("synthetic_b")
        graph.add_link("synthetic_a", "synthetic_b")
        graph.add_link("synthetic_b", "synthetic_a")
        
        assert graph.has_cycle()
    
    def test_detects_indirect_cycle(self):
        """Should detect indirect cycles (A -> B -> C -> A)."""
        graph = ScenarioCausalGraph()
        graph.add_scenario("synthetic_a")
        graph.add_scenario("synthetic_b")
        graph.add_scenario("synthetic_c")
        graph.add_link("synthetic_a", "synthetic_b")
        graph.add_link("synthetic_b", "synthetic_c")
        graph.add_link("synthetic_c", "synthetic_a")
        
        assert graph.has_cycle()
    
    def test_topological_sort_fails_on_cycle(self):
        """Topological sort should raise on cyclic graph."""
        graph = ScenarioCausalGraph()
        graph.add_scenario("synthetic_a")
        graph.add_scenario("synthetic_b")
        graph.add_link("synthetic_a", "synthetic_b")
        graph.add_link("synthetic_b", "synthetic_a")
        
        with pytest.raises(ValueError, match="cycles"):
            graph.topological_sort()


# ==============================================================================
# VISUALIZATION TESTS
# ==============================================================================

class TestVisualization:
    """Tests for graph visualization."""
    
    def test_text_visualization(self):
        """Text visualization should produce readable output."""
        graph = build_default_causal_graph()
        output = visualize_scenario_graph(graph, format="text")
        
        assert SAFETY_LABEL in output
        assert "SCENARIO CAUSAL GRAPH" in output
        assert "Processing Order" in output
    
    def test_dot_visualization(self):
        """DOT visualization should produce valid GraphViz."""
        graph = ScenarioCausalGraph()
        graph.add_scenario("synthetic_test")
        output = visualize_scenario_graph(graph, format="dot")
        
        assert "digraph" in output
        assert "synthetic_test" in output
    
    def test_mermaid_visualization(self):
        """Mermaid visualization should produce valid syntax."""
        graph = ScenarioCausalGraph()
        graph.add_scenario("synthetic_a")
        graph.add_scenario("synthetic_b")
        graph.add_link("synthetic_a", "synthetic_b")
        
        output = visualize_scenario_graph(graph, format="mermaid")
        
        assert "graph TD" in output
        assert "-->" in output
    
    def test_invalid_format_raises(self):
        """Invalid format should raise ValueError."""
        graph = ScenarioCausalGraph()
        
        with pytest.raises(ValueError, match="Unknown format"):
            visualize_scenario_graph(graph, format="invalid")


# ==============================================================================
# REGISTRY INTEGRATION TESTS
# ==============================================================================

class TestRegistryIntegration:
    """Tests for registry integration."""
    
    def test_load_from_registry(self):
        """Should load graph from registry."""
        graph = load_causal_graph_from_registry()
        
        assert len(graph.scenarios) > 0
        assert all(s.startswith("synthetic_") for s in graph.scenarios)
    
    def test_default_graph_has_all_scenarios(self):
        """Default graph should include standard scenarios."""
        graph = build_default_causal_graph()
        
        expected = {
            "synthetic_null_uplift",
            "synthetic_positive_uplift",
            "synthetic_drift_cyclical",
            "synthetic_mixed_chaos",
        }
        
        assert expected.issubset(graph.scenarios)
    
    def test_save_and_reload(self):
        """Graph should survive save and reload."""
        graph = build_default_causal_graph()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"
            save_causal_graph_to_file(graph, path, format="json")
            
            with open(path) as f:
                data = json.load(f)
            
            assert data["label"] == SAFETY_LABEL
            assert len(data["scenarios"]) == len(graph.scenarios)


# ==============================================================================
# VALIDATION TESTS
# ==============================================================================

class TestValidation:
    """Tests for graph validation."""
    
    def test_valid_graph_passes(self):
        """Valid graph should pass validation."""
        graph = build_default_causal_graph()
        errors = graph.validate()
        
        assert len(errors) == 0
    
    def test_invalid_scenario_name_fails(self):
        """Scenario without synthetic_ prefix should fail."""
        graph = ScenarioCausalGraph()
        
        with pytest.raises(ValueError, match="synthetic_"):
            graph.add_scenario("invalid_name")
    
    def test_cyclic_graph_fails_validation(self):
        """Cyclic graph should fail validation."""
        graph = ScenarioCausalGraph()
        graph.add_scenario("synthetic_a")
        graph.add_scenario("synthetic_b")
        graph.add_link("synthetic_a", "synthetic_b")
        graph.add_link("synthetic_b", "synthetic_a")
        
        errors = graph.validate()
        
        assert any("cycle" in e.lower() for e in errors)


# ==============================================================================
# DEPENDENCY TRACKING TESTS
# ==============================================================================

class TestDependencyTracking:
    """Tests for dependency tracking."""
    
    def test_get_dependencies(self):
        """Should return correct dependencies."""
        graph = ScenarioCausalGraph()
        graph.add_scenario("synthetic_a")
        graph.add_scenario("synthetic_b")
        graph.add_scenario("synthetic_c")
        graph.add_link("synthetic_a", "synthetic_c")
        graph.add_link("synthetic_b", "synthetic_c")
        
        deps = graph.get_dependencies("synthetic_c")
        
        assert set(deps) == {"synthetic_a", "synthetic_b"}
    
    def test_get_dependents(self):
        """Should return correct dependents."""
        graph = ScenarioCausalGraph()
        graph.add_scenario("synthetic_a")
        graph.add_scenario("synthetic_b")
        graph.add_scenario("synthetic_c")
        graph.add_link("synthetic_a", "synthetic_b")
        graph.add_link("synthetic_a", "synthetic_c")
        
        deps = graph.get_dependents("synthetic_a")
        
        assert set(deps) == {"synthetic_b", "synthetic_c"}


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

