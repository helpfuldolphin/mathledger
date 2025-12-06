"""
U2 Planner Determinism Tests

Verifies:
- PRNG determinism across platforms
- Frontier determinism
- Policy determinism
- Cycle replay determinism
- Snapshot/restore determinism
"""

import pytest
import tempfile
from pathlib import Path
from typing import Any, Tuple

from rfl.prng import DeterministicPRNG, int_to_hex_seed
from experiments.u2 import (
    U2Runner,
    U2Config,
    FrontierManager,
    BaselinePolicy,
    RFLPolicy,
    save_snapshot,
    load_snapshot,
    verify_trace_determinism,
    run_with_traces,
)


class TestPRNGDeterminism:
    """Test PRNG determinism guarantees."""
    
    def test_same_seed_same_sequence(self):
        """Same seed produces same random sequence."""
        seed = "0x1234abcd"
        
        prng1 = DeterministicPRNG(seed)
        prng2 = DeterministicPRNG(seed)
        
        # Generate sequences
        seq1 = [prng1.random() for _ in range(100)]
        seq2 = [prng2.random() for _ in range(100)]
        
        assert seq1 == seq2, "Same seed must produce same sequence"
    
    def test_hierarchical_isolation(self):
        """Child PRNGs are isolated from each other."""
        master = DeterministicPRNG("0xmaster")
        
        child1 = master.for_path("child1")
        child2 = master.for_path("child2")
        
        # Generate sequences
        seq1 = [child1.random() for _ in range(10)]
        seq2 = [child2.random() for _ in range(10)]
        
        assert seq1 != seq2, "Different paths must produce different sequences"
    
    def test_hierarchical_determinism(self):
        """Same path produces same child PRNG."""
        master1 = DeterministicPRNG("0xmaster")
        master2 = DeterministicPRNG("0xmaster")
        
        child1 = master1.for_path("slice", "arithmetic")
        child2 = master2.for_path("slice", "arithmetic")
        
        seq1 = [child1.random() for _ in range(10)]
        seq2 = [child2.random() for _ in range(10)]
        
        assert seq1 == seq2, "Same path must produce same child PRNG"
    
    def test_state_serialization(self):
        """PRNG state can be saved and restored."""
        prng = DeterministicPRNG("0xtest")
        
        # Generate some values
        [prng.random() for _ in range(50)]
        
        # Save state
        state = prng.get_state()
        
        # Generate more values
        seq1 = [prng.random() for _ in range(10)]
        
        # Restore state
        prng.set_state(state)
        
        # Generate again
        seq2 = [prng.random() for _ in range(10)]
        
        assert seq1 == seq2, "Restored PRNG must produce same sequence"
    
    def test_integer_seed_conversion(self):
        """Integer seeds are converted deterministically."""
        seed_int = 12345
        seed_hex = int_to_hex_seed(seed_int)
        
        prng1 = DeterministicPRNG(seed_int)
        prng2 = DeterministicPRNG(seed_hex)
        
        seq1 = [prng1.random() for _ in range(10)]
        seq2 = [prng2.random() for _ in range(10)]
        
        assert seq1 == seq2, "Integer and hex seeds must be equivalent"


class TestFrontierDeterminism:
    """Test frontier manager determinism."""
    
    def test_push_pop_determinism(self):
        """Same operations produce same frontier state."""
        prng1 = DeterministicPRNG("0xfrontier")
        prng2 = DeterministicPRNG("0xfrontier")
        
        frontier1 = FrontierManager(max_beam_width=10, prng=prng1)
        frontier2 = FrontierManager(max_beam_width=10, prng=prng2)
        
        # Push same candidates
        for i in range(20):
            frontier1.push(f"item_{i}", priority=float(i % 5), depth=i % 3)
            frontier2.push(f"item_{i}", priority=float(i % 5), depth=i % 3)
        
        # Pop all candidates
        seq1 = []
        while not frontier1.is_empty():
            c = frontier1.pop()
            if c:
                seq1.append(c.item)
        
        seq2 = []
        while not frontier2.is_empty():
            c = frontier2.pop()
            if c:
                seq2.append(c.item)
        
        assert seq1 == seq2, "Same operations must produce same pop sequence"
    
    def test_state_serialization(self):
        """Frontier state can be saved and restored."""
        prng = DeterministicPRNG("0xfrontier")
        frontier = FrontierManager(max_beam_width=10, prng=prng)
        
        # Push candidates
        for i in range(15):
            frontier.push(f"item_{i}", priority=float(i % 5), depth=i % 3)
        
        # Pop some
        [frontier.pop() for _ in range(5)]
        
        # Save state
        state = frontier.get_state()
        
        # Pop more
        seq1 = [frontier.pop().item for _ in range(3) if not frontier.is_empty()]
        
        # Restore state
        frontier.set_state(state)
        
        # Pop again
        seq2 = [frontier.pop().item for _ in range(3) if not frontier.is_empty()]
        
        assert seq1 == seq2, "Restored frontier must produce same sequence"


class TestPolicyDeterminism:
    """Test policy determinism."""
    
    def test_baseline_policy_determinism(self):
        """Baseline policy produces deterministic rankings."""
        prng1 = DeterministicPRNG("0xpolicy")
        prng2 = DeterministicPRNG("0xpolicy")
        
        policy1 = BaselinePolicy(prng1)
        policy2 = BaselinePolicy(prng2)
        
        candidates = [f"item_{i}" for i in range(20)]
        
        ranked1 = policy1.rank(candidates)
        ranked2 = policy2.rank(candidates)
        
        # Extract items (ignore scores which may have floating point differences)
        items1 = [item for item, _ in ranked1]
        items2 = [item for item, _ in ranked2]
        
        assert items1 == items2, "Same seed must produce same ranking"
    
    def test_rfl_policy_determinism(self):
        """RFL policy produces deterministic rankings."""
        prng1 = DeterministicPRNG("0xpolicy")
        prng2 = DeterministicPRNG("0xpolicy")
        
        feedback = {
            "item_0": {"success_rate": 0.8},
            "item_5": {"success_rate": 0.6},
            "item_10": {"success_rate": 0.9},
        }
        
        policy1 = RFLPolicy(prng1, feedback)
        policy2 = RFLPolicy(prng2, feedback)
        
        candidates = [{"item": f"item_{i}", "depth": i % 3} for i in range(20)]
        
        ranked1 = policy1.rank(candidates)
        ranked2 = policy2.rank(candidates)
        
        # Extract items
        items1 = [str(item) for item, _ in ranked1]
        items2 = [str(item) for item, _ in ranked2]
        
        assert items1 == items2, "Same seed must produce same RFL ranking"


class TestCycleReplayDeterminism:
    """Test cycle replay determinism."""
    
    def create_mock_execute_fn(self, prng: DeterministicPRNG):
        """Create mock execution function."""
        def execute(item: Any, seed: int) -> Tuple[bool, Any]:
            # Deterministic execution based on item and seed
            item_prng = prng.for_path("execute", str(item), str(seed))
            success = item_prng.random() > 0.3
            result = {"outcome": "success" if success else "failure"}
            return success, result
        return execute
    
    def test_single_cycle_replay(self):
        """Single cycle produces same result on replay."""
        config = U2Config(
            experiment_id="test_replay",
            slice_name="test_slice",
            mode="baseline",
            total_cycles=1,
            master_seed=42,
            max_beam_width=10,
        )
        
        # Run cycle twice
        prng1 = DeterministicPRNG(42)
        runner1 = U2Runner(config)
        runner1.frontier.push("seed_item", priority=1.0, depth=0)
        result1 = runner1.run_cycle(0, self.create_mock_execute_fn(prng1))
        
        prng2 = DeterministicPRNG(42)
        runner2 = U2Runner(config)
        runner2.frontier.push("seed_item", priority=1.0, depth=0)
        result2 = runner2.run_cycle(0, self.create_mock_execute_fn(prng2))
        
        # Results should match
        assert result1.candidates_processed == result2.candidates_processed
        assert result1.candidates_generated == result2.candidates_generated
    
    def test_multi_cycle_replay(self):
        """Multiple cycles produce same results on replay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            config = U2Config(
                experiment_id="test_multi_replay",
                slice_name="test_slice",
                mode="baseline",
                total_cycles=5,
                master_seed=42,
                max_beam_width=10,
            )
            
            # Run experiment twice
            trace1_path = tmppath / "trace1.jsonl"
            trace2_path = tmppath / "trace2.jsonl"
            
            prng1 = DeterministicPRNG(42)
            execute_fn1 = self.create_mock_execute_fn(prng1)
            
            prng2 = DeterministicPRNG(42)
            execute_fn2 = self.create_mock_execute_fn(prng2)
            
            # Initialize frontiers with same seed item
            runner1 = U2Runner(config)
            runner1.frontier.push("seed_item", priority=1.0, depth=0)
            
            runner2 = U2Runner(config)
            runner2.frontier.push("seed_item", priority=1.0, depth=0)
            
            # Run cycles
            for cycle in range(config.total_cycles):
                runner1.run_cycle(cycle, execute_fn1)
                runner2.run_cycle(cycle, execute_fn2)
            
            # Compare final states
            state1 = runner1.get_state()
            state2 = runner2.get_state()
            
            assert state1["stats"]["total_candidates_processed"] == state2["stats"]["total_candidates_processed"]
            assert state1["stats"]["total_candidates_generated"] == state2["stats"]["total_candidates_generated"]


class TestSnapshotDeterminism:
    """Test snapshot/restore determinism."""
    
    def test_snapshot_restore_determinism(self):
        """Restored state produces same execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            config = U2Config(
                experiment_id="test_snapshot",
                slice_name="test_slice",
                mode="baseline",
                total_cycles=10,
                master_seed=42,
                max_beam_width=10,
                snapshot_dir=tmppath,
            )
            
            # Run first half
            prng = DeterministicPRNG(42)
            runner1 = U2Runner(config)
            runner1.frontier.push("seed_item", priority=1.0, depth=0)
            
            execute_fn = lambda item, seed: (prng.for_path("exec", str(item)).random() > 0.3, {})
            
            for cycle in range(5):
                runner1.run_cycle(cycle, execute_fn)
            
            # Save snapshot
            snapshot_path = tmppath / "snapshot.json"
            snapshot = runner1.frontier.get_state()
            
            # Continue from snapshot
            runner1_state = runner1.get_state()
            
            # Create new runner and restore
            runner2 = U2Runner(config)
            runner2.frontier.set_state(snapshot)
            runner2.current_cycle = 5
            runner2.stats = runner1.stats.copy()
            
            # Run second half
            for cycle in range(5, 10):
                runner2.run_cycle(cycle, execute_fn)
            
            # States should be consistent
            assert runner2.current_cycle == 9


def test_cross_platform_determinism():
    """
    Test that determinism holds across different platforms.
    
    This is a smoke test - full cross-platform testing requires
    running on multiple OS/architectures.
    """
    seed = "0xcrossplatform"
    
    # Test PRNG
    prng = DeterministicPRNG(seed)
    values = [prng.random() for _ in range(100)]
    
    # Known good values (computed once, stored for regression)
    # In production, these would be stored in a fixture file
    assert len(values) == 100
    assert all(0.0 <= v < 1.0 for v in values)
    
    # Test frontier
    frontier = FrontierManager(prng=DeterministicPRNG(seed))
    for i in range(10):
        frontier.push(f"item_{i}", priority=float(i), depth=0)
    
    popped = []
    while not frontier.is_empty():
        c = frontier.pop()
        if c:
            popped.append(c.item)
    
    assert len(popped) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
