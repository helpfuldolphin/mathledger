# tests/test_derivation_chain_analysis.py
import unittest
from experiments.derivation_chain_analysis import (
    construct_dependency_dag,
    compute_chain_depth,
    extract_dependency_graph
)

class TestDerivationChainAnalysis(unittest.TestCase):
    """
    PHASE II - NOT USED IN PHASE I
    Unit tests for derivation chain analysis.
    """

    def setUp(self):
        """Set up a sample list of derivations for testing."""
        self.derivations = [
            # Linear chain: A -> B -> C
            {'hash': 'B', 'premises': ['A']},
            {'hash': 'C', 'premises': ['B']},
            # Branched chain: D, E -> F
            {'hash': 'F', 'premises': ['D', 'E']},
            # More complex case: C, F -> G
            {'hash': 'G', 'premises': ['C', 'F']},
            # An isolated derivation
            {'hash': 'I', 'premises': ['H']},
            # An axiom-like derivation (no premises)
            {'hash': 'J', 'premises': []},
        ]
        # Axioms are A, D, E, H.
        
        self.expected_dag = {
            'A': [],
            'B': ['A'],
            'C': ['B'],
            'D': [],
            'E': [],
            'F': ['D', 'E'],
            'G': ['C', 'F'],
            'H': [],
            'I': ['H'],
            'J': [],
        }

    def test_construct_dependency_dag(self):
        """Test the construction of the dependency DAG."""
        dag = construct_dependency_dag(self.derivations)
        self.assertEqual(dag, self.expected_dag)

    def test_construct_dependency_dag_empty(self):
        """Test with an empty list of derivations."""
        dag = construct_dependency_dag([])
        self.assertEqual(dag, {})

    def test_extract_dependency_graph(self):
        """Test extracting dependencies from a single derivation."""
        derivation = {'hash': 'B', 'premises': ['A']}
        conclusion, premises = extract_dependency_graph(derivation)
        self.assertEqual(conclusion, 'B')
        self.assertEqual(premises, ['A'])

        derivation_no_premises = {'hash': 'A', 'premises': []}
        conclusion, premises = extract_dependency_graph(derivation_no_premises)
        self.assertEqual(conclusion, 'A')
        self.assertEqual(premises, [])

    def test_compute_chain_depth(self):
        """Test computation of chain depth for various targets."""
        dag = self.expected_dag
        
        # Axiom depth
        self.assertEqual(compute_chain_depth('A', dag), 1)
        self.assertEqual(compute_chain_depth('D', dag), 1)
        
        # Linear chain depth
        self.assertEqual(compute_chain_depth('B', dag), 2) # A -> B
        self.assertEqual(compute_chain_depth('C', dag), 3) # A -> B -> C
        
        # Branched chain depth
        self.assertEqual(compute_chain_depth('F', dag), 2) # (D or E) -> F
        
        # Complex chain, should take the longest path
        # Path 1 to G: A -> B -> C -> G (depth 4)
        # Path 2 to G: D -> F -> G (depth 3)
        # Path 3 to G: E -> F -> G (depth 3)
        # Max depth should be 4
        self.assertEqual(compute_chain_depth('G', dag), 4)

        # Isolated chain
        self.assertEqual(compute_chain_depth('I', dag), 2) # H -> I

        # Axiom-like derivation with explicit empty premises
        self.assertEqual(compute_chain_depth('J', dag), 1)

    def test_compute_chain_depth_target_not_in_dag(self):
        """Test depth computation for a target not in the DAG."""
        dag = self.expected_dag
        self.assertEqual(compute_chain_depth('Z', dag), 0)

    def test_compute_chain_depth_with_missing_premise_in_dag(self):
        """Test depth computation with an incomplete DAG."""
        # A premise of B ('X') is not a key in the dag.
        dag_incomplete = {'B': ['X'], 'C': ['B']}
        # _get_depth should treat 'X' as an axiom of depth 1.
        # Depth of B should be 1 + depth(X) = 2.
        # Depth of C should be 1 + depth(B) = 3.
        self.assertEqual(compute_chain_depth('C', dag_incomplete), 3)

if __name__ == '__main__':
    unittest.main()
