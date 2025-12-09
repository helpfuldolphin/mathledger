"""
PHASE II — NOT USED IN PHASE I

U2Runner Orchestrator Integration Tests
========================================

Tests that verify U2Runner delegates ordering to cycle_orchestrator,
enforcing INV-RUN-1: No duplication of ordering logic outside cycle_orchestrator.
"""

from __future__ import annotations

import unittest
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch

from experiments.u2.runner import U2Runner, U2Config


class MockPolicy:
    """Mock RFL policy for testing."""
    
    def __init__(self, seed: int):
        self.scores = {}
        self.update_calls: List[Tuple[str, bool]] = []
    
    def score(self, items: List[str]) -> List[float]:
        """Return deterministic scores based on item length."""
        return [float(len(item)) for item in items]
    
    def update(self, item: str, success: bool, failure_signal: Any = None) -> None:
        """Record update calls for verification."""
        self.update_calls.append((item, success))
    
    def get_state(self) -> Tuple:
        return (42,)  # Dummy state
    
    def set_state(self, state: Tuple) -> None:
        pass


def mock_execute_fn(item: str, seed: int) -> Tuple[bool, Any]:
    """Mock execution function that always succeeds."""
    return True, {"item": item, "seed": seed, "outcome": "success"}


def mock_execute_fn_fails(item: str, seed: int) -> Tuple[bool, Any]:
    """Mock execution function that always fails."""
    return False, {"item": item, "seed": seed, "outcome": "failure"}


class TestU2RunnerUsesOrchestrator(unittest.TestCase):
    """Tests that U2Runner delegates to cycle_orchestrator."""

    def test_baseline_mode_uses_get_ordering_strategy(self) -> None:
        """U2Runner.run_cycle should call get_ordering_strategy for baseline."""
        config = U2Config(
            experiment_id="test_baseline",
            slice_name="test_slice",
            mode="baseline",
            total_cycles=5,
            master_seed=42,
        )
        runner = U2Runner(config)
        
        items = ["item_a", "item_b", "item_c"]
        
        with patch("experiments.u2.runner.get_ordering_strategy") as mock_get_strategy:
            # Set up mock strategy
            mock_strategy = MagicMock()
            mock_strategy.order.return_value = ["item_b", "item_a", "item_c"]
            mock_get_strategy.return_value = mock_strategy
            
            result = runner.run_cycle(items, mock_execute_fn)
            
            # Verify get_ordering_strategy was called with correct args
            mock_get_strategy.assert_called_once_with("baseline", None)
            
            # Verify strategy.order was called
            mock_strategy.order.assert_called_once()
            call_args = mock_strategy.order.call_args
            self.assertEqual(call_args[0][0], items)  # items argument
            
            # Verify chosen item matches strategy output
            self.assertEqual(result.item, "item_b")

    def test_rfl_mode_uses_get_ordering_strategy_with_policy(self) -> None:
        """U2Runner.run_cycle should call get_ordering_strategy with policy for RFL."""
        config = U2Config(
            experiment_id="test_rfl",
            slice_name="test_slice",
            mode="rfl",
            total_cycles=5,
            master_seed=42,
        )
        runner = U2Runner(config)
        
        items = ["short", "medium_item", "very_long_item_name"]
        
        with patch("experiments.u2.runner.get_ordering_strategy") as mock_get_strategy:
            mock_strategy = MagicMock()
            mock_strategy.order.return_value = ["very_long_item_name", "medium_item", "short"]
            mock_get_strategy.return_value = mock_strategy
            
            result = runner.run_cycle(items, mock_execute_fn)
            
            # Verify get_ordering_strategy was called with policy
            mock_get_strategy.assert_called_once()
            call_args = mock_get_strategy.call_args
            self.assertEqual(call_args[0][0], "rfl")  # mode
            self.assertIs(call_args[0][1], runner.policy)  # policy object

    def test_no_direct_shuffle_in_baseline(self) -> None:
        """U2Runner should not use random.shuffle directly."""
        import inspect
        from experiments.u2.runner import U2Runner
        
        # Get source code of run_cycle
        source = inspect.getsource(U2Runner.run_cycle)
        
        # Should NOT contain direct shuffle calls
        self.assertNotIn("rng.shuffle", source)
        self.assertNotIn("random.shuffle", source)
        
        # SHOULD contain reference to orchestrator
        self.assertIn("get_ordering_strategy", source)

    def test_no_direct_sorting_in_rfl(self) -> None:
        """U2Runner should not sort items directly for RFL mode."""
        import inspect
        from experiments.u2.runner import U2Runner
        
        source = inspect.getsource(U2Runner.run_cycle)
        
        # Should NOT contain direct sorting on scores
        # (The old pattern was: sorted(zip(items, item_scores), ...))
        self.assertNotIn("sorted(zip(items", source)
        self.assertNotIn("item_scores", source)


class TestU2RunnerOrderingDeterminism(unittest.TestCase):
    """Tests that ordering is deterministic through orchestrator."""

    def test_baseline_ordering_is_deterministic(self) -> None:
        """Same seed should produce same ordering in baseline mode."""
        items = ["a", "b", "c", "d", "e"]
        
        # Run 1
        config1 = U2Config(
            experiment_id="test_1",
            slice_name="test_slice",
            mode="baseline",
            total_cycles=3,
            master_seed=42,
        )
        runner1 = U2Runner(config1)
        results1 = [runner1.run_cycle(items, mock_execute_fn).item for _ in range(3)]
        
        # Run 2 with same seed
        config2 = U2Config(
            experiment_id="test_2",
            slice_name="test_slice",
            mode="baseline",
            total_cycles=3,
            master_seed=42,
        )
        runner2 = U2Runner(config2)
        results2 = [runner2.run_cycle(items, mock_execute_fn).item for _ in range(3)]
        
        self.assertEqual(results1, results2)

    def test_rfl_ordering_is_deterministic(self) -> None:
        """Same seed and policy should produce same ordering in RFL mode."""
        items = ["short", "medium", "longest"]
        
        # Run 1
        config1 = U2Config(
            experiment_id="test_1",
            slice_name="test_slice",
            mode="rfl",
            total_cycles=3,
            master_seed=42,
        )
        runner1 = U2Runner(config1)
        results1 = [runner1.run_cycle(items, mock_execute_fn).item for _ in range(3)]
        
        # Run 2 with same seed
        config2 = U2Config(
            experiment_id="test_2",
            slice_name="test_slice",
            mode="rfl",
            total_cycles=3,
            master_seed=42,
        )
        runner2 = U2Runner(config2)
        results2 = [runner2.run_cycle(items, mock_execute_fn).item for _ in range(3)]
        
        self.assertEqual(results1, results2)


class TestU2RunnerPolicyUpdate(unittest.TestCase):
    """Tests that policy updates still happen after orchestrator refactor."""

    def test_policy_updated_on_success(self) -> None:
        """Policy should be updated after successful execution in RFL mode."""
        config = U2Config(
            experiment_id="test_rfl",
            slice_name="test_slice",
            mode="rfl",
            total_cycles=5,
            master_seed=42,
        )
        runner = U2Runner(config)
        items = ["a", "b", "c"]
        
        initial_update_count = runner.policy_update_count
        
        runner.run_cycle(items, mock_execute_fn)
        
        self.assertEqual(runner.policy_update_count, initial_update_count + 1)

    def test_policy_updated_on_failure(self) -> None:
        """Policy should be updated after failed execution in RFL mode."""
        config = U2Config(
            experiment_id="test_rfl",
            slice_name="test_slice",
            mode="rfl",
            total_cycles=5,
            master_seed=42,
        )
        runner = U2Runner(config)
        items = ["a", "b", "c"]
        
        initial_update_count = runner.policy_update_count
        
        runner.run_cycle(items, mock_execute_fn_fails)
        
        self.assertEqual(runner.policy_update_count, initial_update_count + 1)

    def test_baseline_mode_no_policy_update(self) -> None:
        """Baseline mode should not update any policy."""
        config = U2Config(
            experiment_id="test_baseline",
            slice_name="test_slice",
            mode="baseline",
            total_cycles=5,
            master_seed=42,
        )
        runner = U2Runner(config)
        items = ["a", "b", "c"]
        
        runner.run_cycle(items, mock_execute_fn)
        
        self.assertIsNone(runner.policy)
        self.assertEqual(runner.policy_update_count, 0)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

