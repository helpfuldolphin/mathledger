"""
PHASE II — NOT USED IN PHASE I

Unit tests for experiments.u2.runtime.cycle_orchestrator module.

These tests verify:
    - CycleState and CycleResult dataclasses
    - BaselineOrderingStrategy deterministic shuffle
    - RflOrderingStrategy policy-based ordering
    - execute_cycle function behavior
"""

import unittest
import random
from typing import Any, Dict, List

from experiments.u2.runtime.cycle_orchestrator import (
    CycleState,
    CycleResult,
    OrderingStrategy,
    BaselineOrderingStrategy,
    RflOrderingStrategy,
    execute_cycle,
)


class TestCycleState(unittest.TestCase):
    """Tests for the CycleState dataclass."""

    def test_create_valid_state(self) -> None:
        """Test creating a valid CycleState."""
        state = CycleState(
            cycle=0,
            cycle_seed=42,
            slice_name="test_slice",
            mode="baseline",
            candidate_items=["a", "b", "c"],
        )
        self.assertEqual(state.cycle, 0)
        self.assertEqual(state.cycle_seed, 42)
        self.assertEqual(state.slice_name, "test_slice")
        self.assertEqual(state.mode, "baseline")
        self.assertEqual(state.candidate_items, ["a", "b", "c"])

    def test_rfl_mode_valid(self) -> None:
        """Test creating state with RFL mode."""
        state = CycleState(
            cycle=5,
            cycle_seed=123,
            slice_name="test",
            mode="rfl",
            candidate_items=["x"],
        )
        self.assertEqual(state.mode, "rfl")

    def test_negative_cycle_raises(self) -> None:
        """Test that negative cycle index raises ValueError."""
        with self.assertRaises(ValueError):
            CycleState(
                cycle=-1,
                cycle_seed=42,
                slice_name="test",
                mode="baseline",
                candidate_items=["a"],
            )

    def test_invalid_mode_raises(self) -> None:
        """Test that invalid mode raises ValueError."""
        with self.assertRaises(ValueError):
            CycleState(
                cycle=0,
                cycle_seed=42,
                slice_name="test",
                mode="invalid",
                candidate_items=["a"],
            )

    def test_empty_candidates_raises(self) -> None:
        """Test that empty candidate_items raises ValueError."""
        with self.assertRaises(ValueError):
            CycleState(
                cycle=0,
                cycle_seed=42,
                slice_name="test",
                mode="baseline",
                candidate_items=[],
            )

    def test_immutability(self) -> None:
        """Test that CycleState is immutable."""
        state = CycleState(
            cycle=0,
            cycle_seed=42,
            slice_name="test",
            mode="baseline",
            candidate_items=["a"],
        )
        with self.assertRaises(AttributeError):
            state.cycle = 1


class TestCycleResult(unittest.TestCase):
    """Tests for the CycleResult dataclass."""

    def test_create_success_result(self) -> None:
        """Test creating a successful result."""
        result = CycleResult(
            cycle=0,
            chosen_item="test_item",
            success=True,
            metric_value=1.0,
            raw_result={"outcome": "VERIFIED"},
        )
        self.assertEqual(result.cycle, 0)
        self.assertEqual(result.chosen_item, "test_item")
        self.assertTrue(result.success)
        self.assertEqual(result.metric_value, 1.0)
        self.assertIsNone(result.error_message)

    def test_create_failure_result(self) -> None:
        """Test creating a failed result."""
        result = CycleResult(
            cycle=0,
            chosen_item="test_item",
            success=False,
            metric_value=0.0,
            raw_result={"error": "test error"},
            error_message="Test error occurred",
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Test error occurred")

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        result = CycleResult(
            cycle=0,
            chosen_item="item",
            success=True,
        )
        self.assertEqual(result.metric_value, 0.0)
        self.assertEqual(result.raw_result, {})
        self.assertIsNone(result.error_message)


class TestBaselineOrderingStrategy(unittest.TestCase):
    """Tests for the BaselineOrderingStrategy class."""

    def test_implements_protocol(self) -> None:
        """Test that BaselineOrderingStrategy implements OrderingStrategy."""
        strategy = BaselineOrderingStrategy()
        self.assertTrue(isinstance(strategy, OrderingStrategy))

    def test_deterministic_ordering(self) -> None:
        """Test that same seed produces same ordering."""
        strategy = BaselineOrderingStrategy()
        items = ["a", "b", "c", "d", "e"]

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        result1 = strategy.order(items, rng1)
        result2 = strategy.order(items, rng2)

        self.assertEqual(result1, result2)

    def test_backward_compatibility(self) -> None:
        """
        Test that output matches original inline implementation.

        Original code:
            ordered_items = list(items)
            rng.shuffle(ordered_items)
            chosen_item = ordered_items[0]
        """
        items = ["a", "b", "c", "d", "e"]
        seed = 42

        # New implementation
        strategy = BaselineOrderingStrategy()
        rng_new = random.Random(seed)
        new_result = strategy.order(items, rng_new)

        # Original implementation
        rng_orig = random.Random(seed)
        orig_result = list(items)
        rng_orig.shuffle(orig_result)

        self.assertEqual(new_result, orig_result)

    def test_does_not_modify_input(self) -> None:
        """Test that order() doesn't modify the input list."""
        strategy = BaselineOrderingStrategy()
        items = ["a", "b", "c"]
        original = items.copy()

        rng = random.Random(42)
        strategy.order(items, rng)

        self.assertEqual(items, original)

    def test_different_seeds_produce_different_orderings(self) -> None:
        """Test that different seeds produce different orderings."""
        strategy = BaselineOrderingStrategy()
        items = ["a", "b", "c", "d", "e", "f", "g", "h"]

        rng1 = random.Random(42)
        rng2 = random.Random(43)

        result1 = strategy.order(items, rng1)
        result2 = strategy.order(items, rng2)

        self.assertNotEqual(result1, result2)


class MockPolicy:
    """Mock policy for testing RflOrderingStrategy."""

    def __init__(self, scores: Dict[Any, float]) -> None:
        self._scores = scores

    def score(self, items: List[Any]) -> List[float]:
        return [self._scores.get(item, 0.0) for item in items]


class TestRflOrderingStrategy(unittest.TestCase):
    """Tests for the RflOrderingStrategy class."""

    def test_implements_protocol(self) -> None:
        """Test that RflOrderingStrategy implements OrderingStrategy."""
        policy = MockPolicy({})
        strategy = RflOrderingStrategy(policy)
        self.assertTrue(isinstance(strategy, OrderingStrategy))

    def test_orders_by_score_descending(self) -> None:
        """Test that items are ordered by descending score."""
        policy = MockPolicy({
            "a": 0.1,
            "b": 0.5,
            "c": 0.3,
        })
        strategy = RflOrderingStrategy(policy)
        rng = random.Random(42)  # Not used but required

        result = strategy.order(["a", "b", "c"], rng)

        self.assertEqual(result, ["b", "c", "a"])

    def test_highest_score_first(self) -> None:
        """Test that highest scoring item is first (chosen)."""
        policy = MockPolicy({
            "item1": 0.2,
            "item2": 0.9,
            "item3": 0.5,
        })
        strategy = RflOrderingStrategy(policy)
        rng = random.Random(42)

        result = strategy.order(["item1", "item2", "item3"], rng)

        self.assertEqual(result[0], "item2")

    def test_backward_compatibility(self) -> None:
        """
        Test that output matches original inline implementation.

        Original code:
            item_scores = policy.score(items)
            scored_items = sorted(zip(items, item_scores), key=lambda x: x[1], reverse=True)
            chosen_item = scored_items[0][0]
        """
        scores = {"a": 0.3, "b": 0.7, "c": 0.1}
        items = ["a", "b", "c"]

        # New implementation
        policy = MockPolicy(scores)
        strategy = RflOrderingStrategy(policy)
        rng = random.Random(42)
        new_result = strategy.order(items, rng)

        # Original implementation
        item_scores = [scores[item] for item in items]
        scored_items = sorted(zip(items, item_scores), key=lambda x: x[1], reverse=True)
        orig_result = [item for item, _ in scored_items]

        self.assertEqual(new_result, orig_result)

    def test_rng_not_used(self) -> None:
        """Test that RNG doesn't affect RFL ordering."""
        policy = MockPolicy({"a": 0.1, "b": 0.9})
        strategy = RflOrderingStrategy(policy)

        # Different RNGs should produce same result
        rng1 = random.Random(1)
        rng2 = random.Random(999)

        result1 = strategy.order(["a", "b"], rng1)
        result2 = strategy.order(["a", "b"], rng2)

        self.assertEqual(result1, result2)


class TestExecuteCycle(unittest.TestCase):
    """Tests for the execute_cycle function."""

    def _mock_substrate_success(self, item: str, seed: int) -> Dict[str, Any]:
        """Mock substrate that always succeeds."""
        return {"outcome": "VERIFIED", "item": item, "seed": seed}

    def _mock_substrate_failure(self, item: str, seed: int) -> Dict[str, Any]:
        """Mock substrate that always fails."""
        return {"outcome": "FAILED", "item": item}

    def _mock_substrate_error(self, item: str, seed: int) -> Dict[str, Any]:
        """Mock substrate that raises an exception."""
        raise RuntimeError("Simulated error")

    def test_successful_execution(self) -> None:
        """Test execute_cycle with successful substrate."""
        state = CycleState(
            cycle=0,
            cycle_seed=42,
            slice_name="test",
            mode="baseline",
            candidate_items=["a", "b", "c"],
        )
        strategy = BaselineOrderingStrategy()

        result = execute_cycle(state, strategy, self._mock_substrate_success)

        self.assertEqual(result.cycle, 0)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.chosen_item)
        self.assertIsNone(result.error_message)

    def test_failed_execution(self) -> None:
        """Test execute_cycle with failing substrate."""
        state = CycleState(
            cycle=0,
            cycle_seed=42,
            slice_name="test",
            mode="baseline",
            candidate_items=["a", "b"],
        )
        strategy = BaselineOrderingStrategy()

        result = execute_cycle(state, strategy, self._mock_substrate_failure)

        self.assertFalse(result.success)

    def test_exception_handling(self) -> None:
        """Test execute_cycle handles exceptions gracefully."""
        state = CycleState(
            cycle=0,
            cycle_seed=42,
            slice_name="test",
            mode="baseline",
            candidate_items=["a"],
        )
        strategy = BaselineOrderingStrategy()

        result = execute_cycle(state, strategy, self._mock_substrate_error)

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertIn("error", result.raw_result)

    def test_uses_cycle_seed_for_ordering(self) -> None:
        """Test that execute_cycle uses cycle_seed for deterministic ordering."""
        items = ["a", "b", "c", "d", "e"]

        state1 = CycleState(
            cycle=0,
            cycle_seed=42,
            slice_name="test",
            mode="baseline",
            candidate_items=items,
        )
        state2 = CycleState(
            cycle=0,
            cycle_seed=42,
            slice_name="test",
            mode="baseline",
            candidate_items=items,
        )

        strategy = BaselineOrderingStrategy()

        result1 = execute_cycle(state1, strategy, self._mock_substrate_success)
        result2 = execute_cycle(state2, strategy, self._mock_substrate_success)

        # Same seed should produce same chosen_item
        self.assertEqual(result1.chosen_item, result2.chosen_item)


class TestDeterminismGuarantees(unittest.TestCase):
    """
    Tests ensuring determinism guarantees required by RFL protocol.

    PHASE II — NOT USED IN PHASE I
    """

    def test_baseline_ordering_determinism(self) -> None:
        """Test that baseline ordering is fully deterministic."""
        strategy = BaselineOrderingStrategy()
        items = list("abcdefghij")
        seed = 12345

        results = []
        for _ in range(100):
            rng = random.Random(seed)
            results.append(strategy.order(items, rng))

        self.assertTrue(all(r == results[0] for r in results))

    def test_rfl_ordering_determinism(self) -> None:
        """Test that RFL ordering is deterministic given same scores."""
        policy = MockPolicy({chr(ord('a') + i): i * 0.1 for i in range(10)})
        strategy = RflOrderingStrategy(policy)
        items = list("abcdefghij")

        results = []
        for _ in range(100):
            rng = random.Random(42)
            results.append(strategy.order(items, rng))

        self.assertTrue(all(r == results[0] for r in results))


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

