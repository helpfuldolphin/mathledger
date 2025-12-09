# tests/dag/test_global_dag_builder.py
"""
PHASE II - Unit tests for the GlobalDagBuilder.
"""
import unittest
from pathlib import Path
import json
import tempfile
import warnings

# Add project root for local imports
import sys
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.dag.global_dag_builder import GlobalDagBuilder, DanglingPremiseWarning
from backend.dag.schema import CyclicDependencyError, DanglingPremiseError

class TestGlobalDagBuilder(unittest.TestCase):
    """
    Tests the functionality of the GlobalDagBuilder class.
    """

    def setUp(self):
        # Default builder with no axiom registry
        self.builder = GlobalDagBuilder()
        # Builder with a defined axiom registry
        self.axiom_registry = {"axiom_A", "axiom_B"}
        self.builder_with_axioms = GlobalDagBuilder(axiom_registry=self.axiom_registry)
        # Strict-mode builder
        self.strict_builder = GlobalDagBuilder(axiom_registry=self.axiom_registry, strict=True)


    def test_add_simple_chain(self):
        """Test adding a simple linear chain of derivations."""
        derivations_c0 = [{"conclusion": "A", "premises": []}]
        derivations_c1 = [{"conclusion": "B", "premises": ["A"]}]
        derivations_c2 = [{"conclusion": "C", "premises": ["B"]}]

        self.builder.add_cycle_derivations(0, derivations_c0)
        self.builder.add_cycle_derivations(1, derivations_c1)
        self.builder.add_cycle_derivations(2, derivations_c2)

        self.assertIn("C", self.builder._nodes)
        self.assertIn("B", self.builder._nodes)
        self.assertIn("A", self.builder._nodes)
        
        self.assertEqual(self.builder._global_dag["C"], {("B",)})
        
        # Test metrics
        self.assertEqual(len(self.builder.evolution_metrics), 3)
        metrics_c2 = self.builder.evolution_metrics[2]
        self.assertEqual(metrics_c2["Nodes(t)"], 3)
        self.assertEqual(metrics_c2["ΔNodes(t)"], 1)
        # Edges/derivations are: (A, ()), (B, (A,)), (C, (B,)) -> 3 edges
        self.assertEqual(metrics_c2["Edges(t)"], 3)
        self.assertEqual(metrics_c2["ΔEdges(t)"], 1)
        self.assertEqual(metrics_c2["MaxDepth(t)"], 3)

    def test_multi_proof_support(self):
        """Test that multiple proofs for the same conclusion are stored."""
        derivations = [
            {"conclusion": "C", "premises": ["A"]},
            {"conclusion": "C", "premises": ["B"]},
        ]
        self.builder.add_cycle_derivations(0, derivations)
        
        self.assertEqual(len(self.builder._global_dag["C"]), 2)
        self.assertIn(("A",), self.builder._global_dag["C"])
        self.assertIn(("B",), self.builder._global_dag["C"])
        
        # Test that adding a duplicate proof does not increase the edge count
        self.builder.add_cycle_derivations(1, [{"conclusion": "C", "premises": ["A"]}])
        metrics_c1 = self.builder.evolution_metrics[1]
        self.assertEqual(metrics_c1["Edges(t)"], 2)
        self.assertEqual(metrics_c1["ΔEdges(t)"], 0)
        # Test anomaly hook metric
        self.assertEqual(metrics_c1["total_derivations_in_cycle"], 1)
        self.assertEqual(metrics_c1["duplicate_derivations_in_cycle"], 1)
        self.assertEqual(metrics_c1["new_derivations_in_cycle"], 0)


    def test_acyclicity_detection_longer_cycle(self):
        """Test that a longer cycle is detected correctly."""
        self.builder.add_cycle_derivations(0, [{"conclusion": "C", "premises": ["B"]}])
        self.builder.add_cycle_derivations(1, [{"conclusion": "B", "premises": ["A"]}])
        
        derivations_c2 = [{"conclusion": "A", "premises": ["C"]}]
        with self.assertRaises(CyclicDependencyError) as cm:
            self.builder.add_cycle_derivations(2, derivations_c2)
        
        self.assertEqual(cm.exception.cycle_path, ('C', 'B', 'A', 'C'))

    def test_axiom_registry_integration(self):
        """Test parent existence check with an axiom registry."""
        # This should be fine, premise "axiom_A" is in the registry
        derivations_ok = [{"conclusion": "B", "premises": ["axiom_A"]}]
        self.builder_with_axioms.add_cycle_derivations(0, derivations_ok)
        self.assertIn("B", self.builder_with_axioms._nodes)
        self.assertIn("axiom_A", self.builder_with_axioms._nodes)

    def test_dangling_premise_warning(self):
        """Test that a warning is issued for a dangling premise."""
        # "dangling_premise" is not a known conclusion or axiom
        derivations_bad = [{"conclusion": "C", "premises": ["dangling_premise"]}]
        
        with self.assertWarns(DanglingPremiseWarning) as cm:
            self.builder_with_axioms.add_cycle_derivations(0, derivations_bad)
        
        self.assertIn("not a known conclusion or axiom", str(cm.warning))
        # The node should still be added
        self.assertIn("dangling_premise", self.builder_with_axioms._nodes)

    def test_dangling_premise_error_strict_mode(self):
        """Test that an error is raised for a dangling premise in strict mode."""
        derivations_bad = [{"conclusion": "C", "premises": ["dangling_premise"]}]
        
        with self.assertRaises(DanglingPremiseError) as cm:
            self.strict_builder.add_cycle_derivations(0, derivations_bad)
            
        self.assertEqual(cm.exception.premise, "dangling_premise")
        self.assertEqual(cm.exception.conclusion, "C")

    def test_save_and_load_structure(self):
        """Test saving and loading the DAG structure."""
        derivations = [{"conclusion": "B", "premises": ["A"]}]
        self.builder.add_cycle_derivations(0, derivations)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "structure.json"
            self.builder.save_global_dag_structure(path)
            
            self.assertTrue(path.exists())
            with open(path, "r") as f:
                data = json.load(f)
            
            self.assertIn("B", data)
            self.assertEqual(data["B"], [["A"]])


if __name__ == '__main__':
    unittest.main()
