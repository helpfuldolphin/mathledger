# tests/phase2/metrics/test_metric_integration_replay.py
"""
Phase II Statistical Test Battery - Integration and Replay Tests

Tests for:
1. Cross-function integration scenarios
2. Replay determinism across all metric functions
3. Simulated cycle execution patterns
4. End-to-end metric computation workflows

NO UPLIFT INTERPRETATION: These tests verify mechanical correctness only.
All tests are deterministic and self-contained.
"""

import pytest
import json
from typing import Dict, List, Set, Any, Tuple
from dataclasses import dataclass

from experiments.slice_success_metrics import (
    compute_goal_hit,
    compute_sparse_success,
    compute_chain_success,
    compute_multi_goal_success,
)

from .conftest import (
    DeterministicGenerator,
    SEED_REPLAY,
    assert_tuple_bool_float,
    SLICE_PARAMS,
)


# ===========================================================================
# SIMULATED CYCLE DATA STRUCTURES
# ===========================================================================

@dataclass
class SimulatedCycle:
    """Simulated derivation cycle data for testing."""
    cycle_id: int
    verified_hashes: Set[str]
    derivations: List[Dict[str, Any]]
    candidates_tried: int
    slice_id: str


def generate_simulated_cycles(
    gen: DeterministicGenerator,
    num_cycles: int,
    slice_id: str = "test_slice"
) -> List[SimulatedCycle]:
    """
    Generate deterministic simulated cycle data.
    
    Creates realistic-looking cycle data with:
    - Growing verified hash sets
    - Dependency graphs between derivations
    - Variable candidate counts
    """
    cycles = []
    all_hashes: List[str] = []
    
    for cycle_idx in range(num_cycles):
        # Generate new hashes for this cycle
        new_hash_count = gen.int_value(5, 20)
        new_hashes = [f"h{len(all_hashes) + i}" for i in range(new_hash_count)]
        
        # Create derivations with dependencies to earlier hashes
        derivations = []
        for h in new_hashes:
            premises = []
            if all_hashes and gen.bool_value():
                # Add 1-3 premises from earlier hashes
                num_premises = gen.int_value(1, min(3, len(all_hashes)))
                premises = gen.sample(all_hashes, num_premises)
            derivations.append({"hash": h, "premises": premises})
        
        # Some hashes are verified (80% success rate)
        verified_count = int(new_hash_count * 0.8)
        verified_hashes = set(gen.sample(new_hashes, verified_count))
        
        cycles.append(SimulatedCycle(
            cycle_id=cycle_idx,
            verified_hashes=verified_hashes,
            derivations=derivations,
            candidates_tried=gen.int_value(50, 200),
            slice_id=slice_id
        ))
        
        all_hashes.extend(new_hashes)
    
    return cycles


# ===========================================================================
# INTEGRATION TESTS - COMBINED METRIC SCENARIOS
# ===========================================================================

@pytest.mark.phase2_metrics
class TestMetricIntegration:
    """Tests for integrated metric computation scenarios."""

    def test_all_metrics_same_verified_set(self, gen_replay: DeterministicGenerator):
        """All metric functions work on the same verified set."""
        gen = gen_replay
        gen.reset()
        
        # Build a consistent test dataset
        verified_hashes = gen.hash_set(50)
        target_hashes = gen.hash_set(20)
        required_goals = gen.hash_set(10)
        
        # Build dependency graph
        hash_list = list(verified_hashes)
        dep_graph = {}
        for i in range(10, len(hash_list)):
            dep_graph[hash_list[i]] = [hash_list[gen.int_value(0, i-1)]]
        
        statements = [{"hash": h} for h in verified_hashes]
        
        # Compute all metrics
        goal_result = compute_goal_hit(statements, target_hashes, 10)
        sparse_result = compute_sparse_success(len(verified_hashes), 100, 40)
        
        target_for_chain = hash_list[-1] if hash_list else "h0"
        chain_result = compute_chain_success(statements, dep_graph, target_for_chain, 2)
        multi_result = compute_multi_goal_success(verified_hashes, required_goals)
        
        # All should return valid results
        assert_tuple_bool_float(goal_result)
        assert_tuple_bool_float(sparse_result)
        assert_tuple_bool_float(chain_result)
        assert_tuple_bool_float(multi_result)

    def test_incremental_verification(self, gen_replay: DeterministicGenerator):
        """Simulate incremental verification across cycles."""
        gen = gen_replay
        gen.reset()
        
        accumulated_verified: Set[str] = set()
        target_hashes = {f"target{i}" for i in range(10)}
        
        results = []
        for cycle in range(20):
            # Add some new verified hashes each cycle
            new_verified = gen.hash_set(5)
            # Occasionally verify a target
            if gen.bool_value() and target_hashes:
                new_verified.add(gen.choice(list(target_hashes)))
            
            accumulated_verified.update(new_verified)
            
            statements = [{"hash": h} for h in accumulated_verified]
            result = compute_goal_hit(statements, target_hashes, 5)
            results.append(result)
        
        # Value should be monotonically increasing or stable
        values = [r[1] for r in results]
        for i in range(1, len(values)):
            assert values[i] >= values[i-1], "Goal hits should not decrease"

    def test_dependency_chain_growth(self, gen_replay: DeterministicGenerator):
        """Simulate growing dependency chains."""
        gen = gen_replay
        gen.reset()
        
        dep_graph: Dict[str, List[str]] = {}
        all_hashes: List[str] = ["h0"]  # Root
        verified_hashes: Set[str] = {"h0"}
        
        chain_lengths = []
        
        for cycle in range(30):
            # Extend the chain - always use previous node to ensure linear chain
            new_hash = f"h{cycle + 1}"
            parent = all_hashes[-1]  # Linear chain growth
            dep_graph[new_hash] = [parent]
            all_hashes.append(new_hash)
            
            # Always verify to ensure chain growth
            verified_hashes.add(new_hash)
            
            statements = [{"hash": h} for h in verified_hashes]
            target = all_hashes[-1]
            
            _, chain_len = compute_chain_success(statements, dep_graph, target, 1)
            chain_lengths.append(chain_len)
        
        # With deterministic growth, chain should reach 31 (h0 through h30)
        assert max(chain_lengths) == 31.0, f"Expected chain length 31, got {max(chain_lengths)}"
        # Chain lengths should be monotonically increasing
        for i in range(1, len(chain_lengths)):
            assert chain_lengths[i] >= chain_lengths[i-1], "Chain lengths should not decrease"


# ===========================================================================
# REPLAY DETERMINISM TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.replay
class TestReplayDeterminism:
    """Comprehensive replay determinism tests."""

    def test_goal_hit_replay_100_cycles(self, gen_replay: DeterministicGenerator):
        """goal_hit produces identical results on replay over 100 cycles."""
        gen = gen_replay
        
        # First run
        gen.reset()
        results1 = []
        for _ in range(100):
            statements = gen.statements(gen.int_value(10, 50))
            targets = gen.hash_set(gen.int_value(5, 15))
            min_hits = gen.int_value(1, 5)
            results1.append(compute_goal_hit(statements, targets, min_hits))
        
        # Replay
        gen.reset()
        results2 = []
        for _ in range(100):
            statements = gen.statements(gen.int_value(10, 50))
            targets = gen.hash_set(gen.int_value(5, 15))
            min_hits = gen.int_value(1, 5)
            results2.append(compute_goal_hit(statements, targets, min_hits))
        
        assert results1 == results2

    def test_sparse_success_replay_100_cycles(self, gen_replay: DeterministicGenerator):
        """sparse_success produces identical results on replay over 100 cycles."""
        gen = gen_replay
        
        # First run
        gen.reset()
        results1 = []
        for _ in range(100):
            verified = gen.int_value(0, 100)
            attempted = gen.int_value(verified, 200)
            min_ver = gen.int_value(0, 50)
            results1.append(compute_sparse_success(verified, attempted, min_ver))
        
        # Replay
        gen.reset()
        results2 = []
        for _ in range(100):
            verified = gen.int_value(0, 100)
            attempted = gen.int_value(verified, 200)
            min_ver = gen.int_value(0, 50)
            results2.append(compute_sparse_success(verified, attempted, min_ver))
        
        assert results1 == results2

    def test_chain_success_replay_100_cycles(self, gen_replay: DeterministicGenerator):
        """chain_success produces identical results on replay over 100 cycles."""
        gen = gen_replay
        
        # First run
        gen.reset()
        results1 = []
        for _ in range(100):
            graph, hashes = gen.dependency_graph_linear(gen.int_value(5, 20))
            verified_count = gen.int_value(1, len(hashes))
            statements = gen.statements_from_hashes(hashes[:verified_count])
            target = hashes[gen.int_value(0, len(hashes) - 1)]
            min_len = gen.int_value(1, 5)
            results1.append(compute_chain_success(statements, graph, target, min_len))
        
        # Replay
        gen.reset()
        results2 = []
        for _ in range(100):
            graph, hashes = gen.dependency_graph_linear(gen.int_value(5, 20))
            verified_count = gen.int_value(1, len(hashes))
            statements = gen.statements_from_hashes(hashes[:verified_count])
            target = hashes[gen.int_value(0, len(hashes) - 1)]
            min_len = gen.int_value(1, 5)
            results2.append(compute_chain_success(statements, graph, target, min_len))
        
        assert results1 == results2

    def test_multi_goal_replay_100_cycles(self, gen_replay: DeterministicGenerator):
        """multi_goal produces identical results on replay over 100 cycles."""
        gen = gen_replay
        
        # First run
        gen.reset()
        results1 = []
        for _ in range(100):
            verified = gen.hash_set(gen.int_value(10, 50))
            required = gen.hash_set(gen.int_value(5, 20))
            results1.append(compute_multi_goal_success(verified, required))
        
        # Replay
        gen.reset()
        results2 = []
        for _ in range(100):
            verified = gen.hash_set(gen.int_value(10, 50))
            required = gen.hash_set(gen.int_value(5, 20))
            results2.append(compute_multi_goal_success(verified, required))
        
        assert results1 == results2


# ===========================================================================
# SIMULATED EXPERIMENT REPLAY TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.replay
class TestSimulatedExperimentReplay:
    """Tests simulating full experiment replay scenarios."""

    def test_simulated_50_cycles_replay(self, gen_replay: DeterministicGenerator):
        """50 simulated cycles replay identically."""
        gen = gen_replay
        
        # First run
        gen.reset()
        cycles1 = generate_simulated_cycles(gen, 50)
        
        # Replay
        gen.reset()
        cycles2 = generate_simulated_cycles(gen, 50)
        
        # Verify cycles are identical
        for c1, c2 in zip(cycles1, cycles2):
            assert c1.cycle_id == c2.cycle_id
            assert c1.verified_hashes == c2.verified_hashes
            assert c1.candidates_tried == c2.candidates_tried

    def test_metrics_on_simulated_cycles_replay(self, gen_replay: DeterministicGenerator):
        """Metrics computed on simulated cycles replay identically."""
        gen = gen_replay
        
        # First run
        gen.reset()
        cycles1 = generate_simulated_cycles(gen, 30)
        metrics1 = []
        for cycle in cycles1:
            statements = [{"hash": h} for h in cycle.verified_hashes]
            # Compute sparse success for each cycle
            result = compute_sparse_success(
                len(cycle.verified_hashes),
                cycle.candidates_tried,
                5
            )
            metrics1.append(result)
        
        # Replay
        gen.reset()
        cycles2 = generate_simulated_cycles(gen, 30)
        metrics2 = []
        for cycle in cycles2:
            statements = [{"hash": h} for h in cycle.verified_hashes]
            result = compute_sparse_success(
                len(cycle.verified_hashes),
                cycle.candidates_tried,
                5
            )
            metrics2.append(result)
        
        assert metrics1 == metrics2

    def test_cross_slice_experiment_replay(self, gen_replay: DeterministicGenerator):
        """Multi-slice experiment replays identically."""
        gen = gen_replay
        
        slice_ids = list(SLICE_PARAMS.keys())
        
        # First run
        gen.reset()
        all_results1: Dict[str, List[Tuple[bool, float]]] = {s: [] for s in slice_ids}
        
        for _ in range(20):
            for slice_id in slice_ids:
                verified = gen.hash_set(gen.int_value(10, 30))
                targets = gen.hash_set(gen.int_value(3, 10))
                statements = [{"hash": h} for h in verified]
                result = compute_goal_hit(statements, targets, 2)
                all_results1[slice_id].append(result)
        
        # Replay
        gen.reset()
        all_results2: Dict[str, List[Tuple[bool, float]]] = {s: [] for s in slice_ids}
        
        for _ in range(20):
            for slice_id in slice_ids:
                verified = gen.hash_set(gen.int_value(10, 30))
                targets = gen.hash_set(gen.int_value(3, 10))
                statements = [{"hash": h} for h in verified]
                result = compute_goal_hit(statements, targets, 2)
                all_results2[slice_id].append(result)
        
        assert all_results1 == all_results2


# ===========================================================================
# SERIALIZATION REPLAY TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.replay
class TestSerializationReplay:
    """Tests for JSON serialization and replay."""

    def test_goal_hit_json_roundtrip(self, gen_replay: DeterministicGenerator):
        """goal_hit inputs survive JSON serialization."""
        gen = gen_replay
        gen.reset()
        
        # Generate inputs
        statements = gen.statements(20)
        targets = gen.hash_set(5)
        min_hits = 3
        
        # Serialize to JSON
        data = {
            "statements": statements,
            "targets": list(targets),
            "min_hits": min_hits
        }
        json_str = json.dumps(data)
        
        # Deserialize
        loaded = json.loads(json_str)
        loaded_statements = loaded["statements"]
        loaded_targets = set(loaded["targets"])
        loaded_min_hits = loaded["min_hits"]
        
        # Compute on original
        result1 = compute_goal_hit(statements, targets, min_hits)
        
        # Compute on deserialized
        result2 = compute_goal_hit(loaded_statements, loaded_targets, loaded_min_hits)
        
        assert result1 == result2

    def test_chain_length_json_roundtrip(self, gen_replay: DeterministicGenerator):
        """chain_length inputs survive JSON serialization."""
        gen = gen_replay
        gen.reset()
        
        graph, hashes = gen.dependency_graph_linear(15)
        statements = gen.statements_from_hashes(hashes)
        target = hashes[-1]
        min_len = 5
        
        # Serialize
        data = {
            "statements": statements,
            "graph": graph,
            "target": target,
            "min_len": min_len
        }
        json_str = json.dumps(data)
        
        # Deserialize
        loaded = json.loads(json_str)
        
        # Compute on original
        result1 = compute_chain_success(statements, graph, target, min_len)
        
        # Compute on deserialized
        result2 = compute_chain_success(
            loaded["statements"],
            loaded["graph"],
            loaded["target"],
            loaded["min_len"]
        )
        
        assert result1 == result2


# ===========================================================================
# STRESS TESTS - LARGE SCALE REPLAY
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.large_scale
@pytest.mark.replay
class TestLargeScaleReplay:
    """Large scale replay tests."""

    def test_1000_goal_hit_replay(self, gen_replay: DeterministicGenerator):
        """1000 goal_hit computations replay identically."""
        gen = gen_replay
        
        # First run
        gen.reset()
        results1 = []
        for _ in range(1000):
            statements = gen.statements(gen.int_value(5, 25))
            targets = gen.hash_set(gen.int_value(2, 8))
            min_hits = gen.int_value(1, 3)
            results1.append(compute_goal_hit(statements, targets, min_hits))
        
        # Replay
        gen.reset()
        results2 = []
        for _ in range(1000):
            statements = gen.statements(gen.int_value(5, 25))
            targets = gen.hash_set(gen.int_value(2, 8))
            min_hits = gen.int_value(1, 3)
            results2.append(compute_goal_hit(statements, targets, min_hits))
        
        assert results1 == results2

    def test_500_mixed_metrics_replay(self, gen_replay: DeterministicGenerator):
        """500 mixed metric computations replay identically."""
        gen = gen_replay
        metric_funcs = ["goal_hit", "sparse", "chain", "multi"]
        
        def compute_random_metric(gen: DeterministicGenerator, func_name: str):
            if func_name == "goal_hit":
                return compute_goal_hit(
                    gen.statements(gen.int_value(10, 30)),
                    gen.hash_set(gen.int_value(3, 10)),
                    gen.int_value(1, 5)
                )
            elif func_name == "sparse":
                return compute_sparse_success(
                    gen.int_value(0, 50),
                    gen.int_value(50, 150),
                    gen.int_value(0, 30)
                )
            elif func_name == "chain":
                graph, hashes = gen.dependency_graph_linear(gen.int_value(5, 15))
                return compute_chain_success(
                    gen.statements_from_hashes(hashes),
                    graph,
                    hashes[-1] if hashes else "h0",
                    gen.int_value(1, 5)
                )
            else:  # multi
                return compute_multi_goal_success(
                    gen.hash_set(gen.int_value(10, 30)),
                    gen.hash_set(gen.int_value(3, 10))
                )
        
        # First run
        gen.reset()
        results1 = []
        for _ in range(500):
            func_name = gen.choice(metric_funcs)
            results1.append(compute_random_metric(gen, func_name))
        
        # Replay
        gen.reset()
        results2 = []
        for _ in range(500):
            func_name = gen.choice(metric_funcs)
            results2.append(compute_random_metric(gen, func_name))
        
        assert results1 == results2


# ===========================================================================
# CROSS-FUNCTION CONSISTENCY TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
class TestCrossFunctionConsistency:
    """Tests for consistency across metric functions."""

    def test_goal_hit_and_multi_goal_consistency(self):
        """goal_hit and multi_goal agree on intersection size."""
        verified = {"h1", "h2", "h3", "h4", "h5"}
        targets = {"h1", "h3", "h6"}
        
        statements = [{"hash": h} for h in verified]
        
        # goal_hit counts intersection (when used with min_hits=0)
        _, gh_value = compute_goal_hit(statements, targets, 0)
        
        # multi_goal also counts intersection
        _, mg_value = compute_multi_goal_success(verified, targets)
        
        # Both should count the same intersection
        assert gh_value == mg_value

    def test_sparse_success_identity(self):
        """sparse_success value equals input verified_count."""
        for v in [0, 1, 10, 50, 100]:
            _, value = compute_sparse_success(v, 200, 0)
            assert value == float(v)

    def test_chain_length_single_node(self):
        """chain_length of isolated verified node is 1."""
        statements = [{"hash": "isolated"}]
        _, value = compute_chain_success(statements, {}, "isolated", 0)
        assert value == 1.0


# ===========================================================================
# EDGE CASE INTEGRATION TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.degenerate
class TestEdgeCaseIntegration:
    """Integration tests for edge cases."""

    def test_all_metrics_on_empty_data(self):
        """All metrics handle completely empty data."""
        empty_statements: List[Dict[str, Any]] = []
        empty_set: Set[str] = set()
        empty_graph: Dict[str, List[str]] = {}
        
        gh = compute_goal_hit(empty_statements, empty_set, 0)
        ss = compute_sparse_success(0, 0, 0)
        cs = compute_chain_success(empty_statements, empty_graph, "h0", 0)
        mg = compute_multi_goal_success(empty_set, empty_set)
        
        # All should succeed with min=0
        assert gh == (True, 0.0)
        assert ss == (True, 0.0)
        assert cs == (True, 0.0)
        assert mg == (True, 0.0)

    def test_all_metrics_on_single_element(self):
        """All metrics handle single element data."""
        single_statements = [{"hash": "only"}]
        single_set = {"only"}
        
        gh = compute_goal_hit(single_statements, single_set, 1)
        ss = compute_sparse_success(1, 1, 1)
        cs = compute_chain_success(single_statements, {}, "only", 1)
        mg = compute_multi_goal_success(single_set, single_set)
        
        # All should succeed
        assert gh == (True, 1.0)
        assert ss == (True, 1.0)
        assert cs == (True, 1.0)
        assert mg == (True, 1.0)

    def test_metrics_with_very_large_threshold(self):
        """Metrics fail gracefully with impossibly large thresholds."""
        statements = [{"hash": "h1"}]
        verified_set = {"h1"}
        
        gh = compute_goal_hit(statements, verified_set, 1000000)
        ss = compute_sparse_success(1, 10, 1000000)
        cs = compute_chain_success(statements, {}, "h1", 1000000)
        # multi_goal: success requires all goals met, not a threshold
        
        assert gh[0] is False
        assert ss[0] is False
        assert cs[0] is False


# ===========================================================================
# DETERMINISM CONTRACT TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.determinism
class TestDeterminismContract:
    """Tests verifying the determinism contract across all metrics."""

    def test_no_side_effects(self, gen_replay: DeterministicGenerator):
        """Metric functions have no observable side effects."""
        gen = gen_replay
        gen.reset()
        
        # Create mutable inputs
        statements = gen.statements(20)
        targets = gen.hash_set(10)
        verified = gen.hash_set(15)
        graph, hashes = gen.dependency_graph_linear(10)
        
        # Copy originals
        statements_copy = [dict(s) for s in statements]
        targets_copy = set(targets)
        verified_copy = set(verified)
        graph_copy = dict(graph)
        
        # Call all functions
        compute_goal_hit(statements, targets, 5)
        compute_sparse_success(len(verified), 100, 5)
        compute_chain_success(gen.statements_from_hashes(hashes), graph, hashes[-1], 3)
        compute_multi_goal_success(verified, targets)
        
        # Verify inputs unchanged
        assert statements == statements_copy
        assert targets == targets_copy
        assert verified == verified_copy
        assert graph == graph_copy

    def test_thread_safety_simulation(self, gen_replay: DeterministicGenerator):
        """Simulate thread-safe execution (sequential version)."""
        gen = gen_replay
        
        # Generate a batch of test cases
        gen.reset()
        test_cases = []
        for _ in range(50):
            statements = gen.statements(gen.int_value(10, 30))
            targets = gen.hash_set(gen.int_value(3, 10))
            min_hits = gen.int_value(1, 5)
            test_cases.append((statements, targets, min_hits))
        
        # Run in "parallel" (sequential simulation)
        results1 = [compute_goal_hit(*case) for case in test_cases]
        
        # Run again in different order
        reversed_cases = list(reversed(test_cases))
        results2 = [compute_goal_hit(*case) for case in reversed_cases]
        
        # Results should match regardless of order
        for i, case in enumerate(test_cases):
            j = len(test_cases) - 1 - i  # Reverse index
            assert results1[i] == results2[j]

