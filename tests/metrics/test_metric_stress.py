"""
PHASE II — NOT USED IN PHASE I
Metric Engine Integration Stress Tests

Based on METRIC_INTEGRATION_CONSISTENCY_SPEC.md.
"""
import unittest
import random
import sys
from typing import Any, Dict, List, Set

# Ensure experiments module is discoverable
sys.path.insert(0, '.')
from experiments.derivation_chain_analysis import ChainAnalyzer

def generate_synthetic_derivations(
    num_nodes: int,
    max_fan_out: int = 3,
    cycle_prob: float = 0.0,
    dangling_prob: float = 0.0,
    malformed_prob: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Generates a list of synthetic derivation objects for stress testing.
    """
    derivations = []
    nodes = [f"h_{i}" for i in range(num_nodes)]
    
    for i, node_hash in enumerate(nodes):
        if random.random() < malformed_prob:
            # Create a malformed entry
            if random.random() < 0.5:
                derivations.append({"premises": []}) # Missing hash
            else:
                derivations.append({"hash": node_hash}) # Missing premises
            continue

        # Determine premises
        premises = []
        if i > 0: # Node 0 is always an axiom
            num_premises = random.randint(1, max_fan_out)
            # Potential parents are all nodes before the current one
            potential_parents = nodes[:i]
            for _ in range(num_premises):
                if random.random() < dangling_prob:
                    premises.append(f"dangling_{random.randint(0, 1e6)}")
                    continue
                
                # Add a cycle?
                if i > 1 and random.random() < cycle_prob:
                     # Link back to a random node (could be a grandparent etc)
                    premise_candidate = random.choice(nodes[:i-1])
                else:
                    premise_candidate = random.choice(potential_parents)

                if premise_candidate not in premises:
                    premises.append(premise_candidate)

        derivations.append({"hash": node_hash, "premises": premises})
        
    return derivations

class TestMetricStress(unittest.TestCase):

    def test_max_recursion_depth(self):
        """Tests against Python's recursion limit with a very deep chain."""
        print("\nRunning test_max_recursion_depth...")
        # Keep this number high but reasonable to avoid excessively long tests.
        # Python's default limit is ~1000.
        depth = 900
        derivations = [{"hash": "h_0", "premises": []}]
        for i in range(1, depth):
            derivations.append({"hash": f"h_{i}", "premises": [f"h_{i-1}"]})

        try:
            analyzer = ChainAnalyzer(derivations)
            # This will raise RecursionError if the implementation is not robust.
            # Our memoized implementation should handle this fine.
            calculated_depth = analyzer.get_depth(f"h_{depth-1}")
            self.assertEqual(calculated_depth, depth)
        except RecursionError:
            self.fail("ChainAnalyzer failed due to Python's recursion limit.")

    def test_cycle_termination(self):
        """Tests that get_depth terminates on graphs with cycles."""
        print("\nRunning test_cycle_termination...")
        # h0 <- h1 <- h2 <- h0
        derivations = [
            {"hash": "h0", "premises": ["h2"]},
            {"hash": "h1", "premises": ["h0"]},
            {"hash": "h2", "premises": ["h1"]},
            {"hash": "h3", "premises": ["h2"]}, # A sane node
        ]
        analyzer = ChainAnalyzer(derivations)
        try:
            # The exact depth can be implementation-dependent, but it must not hang.
            # We expect our implementation to return a stable value.
            depth = analyzer.get_depth("h3")
            self.assertIsInstance(depth, int)
            self.assertGreater(depth, 0)
        except RecursionError:
            self.fail("ChainAnalyzer failed to terminate on a cyclic graph.")

    def test_dangling_edge_behavior(self):
        """Tests that dangling premises are treated as having depth 1."""
        print("\nRunning test_dangling_edge_behavior...")
        # h1 depends on a hash that doesn't exist
        derivations = [
            {"hash": "h0", "premises": []},
            {"hash": "h1", "premises": ["dangling_premise"]},
            {"hash": "h2", "premises": ["h1"]},
        ]
        analyzer = ChainAnalyzer(derivations)
        
        # Per spec, depth(dangling_premise) = 1
        # depth(h1) = 1 + depth(dangling_premise) = 2
        self.assertEqual(analyzer.get_depth("h1"), 2)
        # depth(h2) = 1 + depth(h1) = 3
        self.assertEqual(analyzer.get_depth("h2"), 3)

    def test_malformed_log_resilience(self):
        """Tests that the system is resilient to malformed data."""
        print("\nRunning test_malformed_log_resilience...")
        # This log is missing a 'hash' key
        malformed_derivations = [{"premises": []}]
        try:
            # Analyzer should not crash on init
            analyzer = ChainAnalyzer(malformed_derivations)
            # Its internal graph should be empty as it skipped the bad entry
            self.assertEqual(len(analyzer._dep_graph), 0)
        except Exception as e:
            self.fail(f"ChainAnalyzer crashed on malformed data (missing hash): {e}")

        # This log is missing a 'premises' key
        malformed_derivations = [{"hash": "h0"}]
        try:
            # Analyzer should not crash
            analyzer = ChainAnalyzer(malformed_derivations)
            # It should create the node but with no premises
            self.assertEqual(analyzer._dep_graph["h0"], [])
        except Exception as e:
            self.fail(f"ChainAnalyzer crashed on malformed data (missing premises): {e}")

if __name__ == '__main__':
    # This allows running the stress tests directly.
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMetricStress))
    runner = unittest.TextTestRunner()
    runner.run(suite)
