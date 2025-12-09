# tests/phase2/metrics_adversarial/test_extended_replay.py
"""
Extended Replay Tests (10,000 iterations)

Confirms no entropy leaks by verifying:
- 10,000-iteration replay determinism
- Cross-seed independence
- No hidden state accumulation
- Memory-stable batch processing

NO METRIC INTERPRETATION: These tests verify determinism only.
"""

import random
import pytest
from typing import Dict, List, Set, Any, Tuple

from backend.substrate.slice_success_metrics import (
    compute_goal_hit,
    compute_sparse_success,
    compute_chain_success,
    compute_multi_goal_success,
)

from .conftest import (
    BatchGenerator,
    SEED_REPLAY_EXTENDED,
    SEED_BATCH,
)


# ===========================================================================
# 10,000 ITERATION REPLAY TESTS
# ===========================================================================

@pytest.mark.entropy
@pytest.mark.high_volume
class TestExtendedReplay10K:
    """10,000-iteration replay tests to detect entropy leaks."""

    def test_goal_hit_10k_replay(self):
        """goal_hit produces identical results over 10,000 replays."""
        rng = random.Random(SEED_REPLAY_EXTENDED)
        
        # First run
        results1 = []
        for _ in range(10000):
            num_stmts = rng.randint(5, 30)
            num_targets = rng.randint(2, 10)
            
            statements = [{"hash": f"h{rng.randint(0, 999)}"} for _ in range(num_stmts)]
            targets = {f"h{rng.randint(0, 999)}" for _ in range(num_targets)}
            min_hits = rng.randint(0, min(5, num_targets))
            
            result = compute_goal_hit(statements, targets, min_hits)
            results1.append(result)
        
        # Replay
        rng = random.Random(SEED_REPLAY_EXTENDED)
        results2 = []
        for _ in range(10000):
            num_stmts = rng.randint(5, 30)
            num_targets = rng.randint(2, 10)
            
            statements = [{"hash": f"h{rng.randint(0, 999)}"} for _ in range(num_stmts)]
            targets = {f"h{rng.randint(0, 999)}" for _ in range(num_targets)}
            min_hits = rng.randint(0, min(5, num_targets))
            
            result = compute_goal_hit(statements, targets, min_hits)
            results2.append(result)
        
        assert results1 == results2, "Entropy leak detected in goal_hit"

    def test_sparse_success_10k_replay(self):
        """sparse_success produces identical results over 10,000 replays."""
        rng = random.Random(SEED_REPLAY_EXTENDED)
        
        # First run
        results1 = []
        for _ in range(10000):
            verified = rng.randint(0, 1000)
            attempted = rng.randint(verified, 2000)
            min_ver = rng.randint(0, 500)
            
            result = compute_sparse_success(verified, attempted, min_ver)
            results1.append(result)
        
        # Replay
        rng = random.Random(SEED_REPLAY_EXTENDED)
        results2 = []
        for _ in range(10000):
            verified = rng.randint(0, 1000)
            attempted = rng.randint(verified, 2000)
            min_ver = rng.randint(0, 500)
            
            result = compute_sparse_success(verified, attempted, min_ver)
            results2.append(result)
        
        assert results1 == results2, "Entropy leak detected in sparse_success"

    def test_multi_goal_10k_replay(self):
        """multi_goal produces identical results over 10,000 replays."""
        rng = random.Random(SEED_REPLAY_EXTENDED)
        
        # First run
        results1 = []
        for _ in range(10000):
            num_verified = rng.randint(5, 50)
            num_required = rng.randint(2, 20)
            
            verified = {f"h{rng.randint(0, 999)}" for _ in range(num_verified)}
            required = {f"h{rng.randint(0, 999)}" for _ in range(num_required)}
            
            result = compute_multi_goal_success(verified, required)
            results1.append(result)
        
        # Replay
        rng = random.Random(SEED_REPLAY_EXTENDED)
        results2 = []
        for _ in range(10000):
            num_verified = rng.randint(5, 50)
            num_required = rng.randint(2, 20)
            
            verified = {f"h{rng.randint(0, 999)}" for _ in range(num_verified)}
            required = {f"h{rng.randint(0, 999)}" for _ in range(num_required)}
            
            result = compute_multi_goal_success(verified, required)
            results2.append(result)
        
        assert results1 == results2, "Entropy leak detected in multi_goal"

    def test_chain_success_10k_replay_shallow(self):
        """chain_success produces identical results over 10,000 shallow chain replays."""
        rng = random.Random(SEED_REPLAY_EXTENDED)
        
        # First run (shallow chains to avoid recursion limits)
        results1 = []
        for _ in range(10000):
            depth = rng.randint(2, 10)
            hashes = [f"h{i}" for i in range(depth)]
            graph = {hashes[i]: [hashes[i-1]] for i in range(1, depth)}
            statements = [{"hash": h} for h in hashes]
            target = hashes[-1]
            min_len = rng.randint(1, depth)
            
            result = compute_chain_success(statements, graph, target, min_len)
            results1.append(result)
        
        # Replay
        rng = random.Random(SEED_REPLAY_EXTENDED)
        results2 = []
        for _ in range(10000):
            depth = rng.randint(2, 10)
            hashes = [f"h{i}" for i in range(depth)]
            graph = {hashes[i]: [hashes[i-1]] for i in range(1, depth)}
            statements = [{"hash": h} for h in hashes]
            target = hashes[-1]
            min_len = rng.randint(1, depth)
            
            result = compute_chain_success(statements, graph, target, min_len)
            results2.append(result)
        
        assert results1 == results2, "Entropy leak detected in chain_success"


# ===========================================================================
# CROSS-SEED INDEPENDENCE TESTS
# ===========================================================================

@pytest.mark.entropy
class TestCrossSeedIndependence:
    """Tests verifying different seeds produce different results."""

    def test_different_seeds_different_inputs(self):
        """Different seeds generate different test inputs."""
        seed1 = SEED_REPLAY_EXTENDED
        seed2 = SEED_REPLAY_EXTENDED + 1
        
        rng1 = random.Random(seed1)
        rng2 = random.Random(seed2)
        
        inputs1 = []
        inputs2 = []
        
        for _ in range(100):
            inputs1.append(rng1.randint(0, 1000))
            inputs2.append(rng2.randint(0, 1000))
        
        # Inputs should differ
        assert inputs1 != inputs2

    def test_same_seed_same_results_across_runs(self):
        """Same seed produces identical results in separate test runs."""
        seed = SEED_REPLAY_EXTENDED
        
        def run_computation(seed: int) -> List[Tuple[bool, float]]:
            rng = random.Random(seed)
            results = []
            for _ in range(1000):
                verified = rng.randint(0, 100)
                attempted = rng.randint(100, 200)
                min_ver = rng.randint(0, 50)
                results.append(compute_sparse_success(verified, attempted, min_ver))
            return results
        
        run1 = run_computation(seed)
        run2 = run_computation(seed)
        run3 = run_computation(seed)
        
        assert run1 == run2 == run3


# ===========================================================================
# NO HIDDEN STATE TESTS
# ===========================================================================

@pytest.mark.entropy
class TestNoHiddenState:
    """Tests verifying no hidden state accumulation."""

    def test_goal_hit_no_state_accumulation(self):
        """goal_hit doesn't accumulate state between calls."""
        statements = [{"hash": "h1"}, {"hash": "h2"}]
        targets = {"h1", "h2"}
        
        # Make many calls
        for _ in range(1000):
            result = compute_goal_hit(statements, targets, 2)
        
        # Final result should be same as first
        final_result = compute_goal_hit(statements, targets, 2)
        assert final_result == (True, 2.0)

    def test_sparse_success_no_state_accumulation(self):
        """sparse_success doesn't accumulate state between calls."""
        # Make many calls with different inputs
        for i in range(1000):
            compute_sparse_success(i % 100, 200, 50)
        
        # Standard call should work normally
        result = compute_sparse_success(75, 100, 50)
        assert result == (True, 75.0)

    def test_chain_success_no_state_accumulation(self):
        """chain_success doesn't accumulate state between calls."""
        # Different graphs
        for i in range(100):
            graph = {f"h{i}": [f"h{i-1}"]} if i > 0 else {}
            statements = [{"hash": f"h{j}"} for j in range(i + 1)]
            compute_chain_success(statements, graph, f"h{i}", 1)
        
        # Standard call should work normally
        statements = [{"hash": "h0"}, {"hash": "h1"}]
        graph = {"h1": ["h0"]}
        result = compute_chain_success(statements, graph, "h1", 2)
        assert result == (True, 2.0)

    def test_multi_goal_no_state_accumulation(self):
        """multi_goal doesn't accumulate state between calls."""
        # Make many calls
        for i in range(1000):
            verified = {f"h{j}" for j in range(i % 50)}
            required = {f"h{j}" for j in range(i % 20)}
            compute_multi_goal_success(verified, required)
        
        # Standard call should work normally
        result = compute_multi_goal_success({"h1", "h2"}, {"h1"})
        assert result == (True, 1.0)


# ===========================================================================
# HIGH-VOLUME BATCH TESTS (10^5 CYCLES)
# ===========================================================================

@pytest.mark.entropy
@pytest.mark.high_volume
class TestHighVolumeBatch:
    """High-volume batch tests (10^5 cycles)."""

    def test_goal_hit_100k_batch_determinism(self, batch_generator: BatchGenerator):
        """100,000 goal_hit computations are deterministic."""
        batch_generator.reset()
        batch = batch_generator.generate_goal_hit_batch(100000, max_statements=20, max_targets=10)
        
        # Compute results
        results = []
        for statements, targets, min_hits in batch:
            result = compute_goal_hit(statements, targets, min_hits)
            results.append(result)
        
        # Replay
        batch_generator.reset()
        batch2 = batch_generator.generate_goal_hit_batch(100000, max_statements=20, max_targets=10)
        
        results2 = []
        for statements, targets, min_hits in batch2:
            result = compute_goal_hit(statements, targets, min_hits)
            results2.append(result)
        
        assert results == results2, "100k batch not deterministic"

    def test_sparse_success_100k_batch_determinism(self, batch_generator: BatchGenerator):
        """100,000 sparse_success computations are deterministic."""
        batch_generator.reset()
        batch = batch_generator.generate_sparse_success_batch(100000)
        
        results = [compute_sparse_success(v, a, m) for v, a, m in batch]
        
        batch_generator.reset()
        batch2 = batch_generator.generate_sparse_success_batch(100000)
        
        results2 = [compute_sparse_success(v, a, m) for v, a, m in batch2]
        
        assert results == results2

    def test_multi_goal_100k_batch_determinism(self, batch_generator: BatchGenerator):
        """100,000 multi_goal computations are deterministic."""
        batch_generator.reset()
        batch = batch_generator.generate_multi_goal_batch(100000)
        
        results = [compute_multi_goal_success(v, r) for v, r in batch]
        
        batch_generator.reset()
        batch2 = batch_generator.generate_multi_goal_batch(100000)
        
        results2 = [compute_multi_goal_success(v, r) for v, r in batch2]
        
        assert results == results2


# ===========================================================================
# STATISTICAL DISTRIBUTION TESTS
# ===========================================================================

@pytest.mark.entropy
class TestStatisticalDistribution:
    """Tests verifying expected distributions in random inputs."""

    def test_goal_hit_success_rate_stable(self, batch_generator: BatchGenerator):
        """goal_hit success rate is stable across replays."""
        def compute_success_rate(seed: int, n: int) -> float:
            gen = BatchGenerator(seed)
            batch = gen.generate_goal_hit_batch(n)
            
            successes = 0
            for statements, targets, min_hits in batch:
                success, _ = compute_goal_hit(statements, targets, min_hits)
                if success:
                    successes += 1
            
            return successes / n
        
        # Same seed should give same success rate
        rate1 = compute_success_rate(SEED_BATCH, 10000)
        rate2 = compute_success_rate(SEED_BATCH, 10000)
        
        assert rate1 == rate2

    def test_sparse_success_boundary_count_stable(self, batch_generator: BatchGenerator):
        """sparse_success boundary hit count is stable."""
        def count_boundaries(seed: int, n: int) -> int:
            gen = BatchGenerator(seed)
            batch = gen.generate_sparse_success_batch(n)
            
            boundaries = 0
            for verified, attempted, min_ver in batch:
                if verified == min_ver:
                    boundaries += 1
            
            return boundaries
        
        count1 = count_boundaries(SEED_BATCH, 10000)
        count2 = count_boundaries(SEED_BATCH, 10000)
        
        assert count1 == count2


# ===========================================================================
# INCREMENTAL REPLAY TESTS
# ===========================================================================

@pytest.mark.entropy
class TestIncrementalReplay:
    """Tests verifying incremental replay consistency."""

    def test_prefix_replay_consistency(self):
        """First N results of M-run match N-run."""
        seed = SEED_REPLAY_EXTENDED
        
        # 1000-run
        rng1000 = random.Random(seed)
        results_1000 = []
        for _ in range(1000):
            v = rng1000.randint(0, 100)
            results_1000.append(compute_sparse_success(v, 200, 50))
        
        # 500-run (prefix)
        rng500 = random.Random(seed)
        results_500 = []
        for _ in range(500):
            v = rng500.randint(0, 100)
            results_500.append(compute_sparse_success(v, 200, 50))
        
        # First 500 of 1000-run should match 500-run
        assert results_1000[:500] == results_500

    def test_suffix_independent_of_prefix(self):
        """Later results depend only on seed state, not computation results."""
        seed = SEED_REPLAY_EXTENDED
        
        # Run 1: compute first 100, then next 100
        rng = random.Random(seed)
        for _ in range(100):
            rng.randint(0, 100)  # Consume state
        
        results_second_100 = []
        for _ in range(100):
            v = rng.randint(0, 100)
            results_second_100.append(compute_sparse_success(v, 200, 50))
        
        # Run 2: skip first 100, compute next 100
        rng = random.Random(seed)
        for _ in range(100):
            rng.randint(0, 100)  # Same state consumption
        
        results_second_100_replay = []
        for _ in range(100):
            v = rng.randint(0, 100)
            results_second_100_replay.append(compute_sparse_success(v, 200, 50))
        
        assert results_second_100 == results_second_100_replay

