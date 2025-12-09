"""
PHASE II — NOT USED IN PHASE I

Unit tests for cycle_orchestrator input validation guards.

Tests the early validation of invalid inputs with clear error messages.
"""

import random
import unittest
from typing import Any, Dict

from experiments.u2.runtime.cycle_orchestrator import (
    CycleState,
    CycleResult,
    CycleExecutionError,
    BaselineOrderingStrategy,
    RflOrderingStrategy,
    execute_cycle,
    get_ordering_strategy,
)
from experiments.u2.runtime.error_classifier import ErrorContext, RuntimeErrorKind


class TestCycleStateValidation(unittest.TestCase):
    """Tests for CycleState input validation."""

    def test_negative_cycle_raises(self) -> None:
        """Test that negative cycle raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CycleState(
                cycle=-1,
                cycle_seed=42,
                slice_name="test",
                mode="baseline",
                candidate_items=["a"],
            )
        self.assertIn("non-negative", str(ctx.exception))

    def test_empty_slice_name_raises(self) -> None:
        """Test that empty slice_name raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CycleState(
                cycle=0,
                cycle_seed=42,
                slice_name="",
                mode="baseline",
                candidate_items=["a"],
            )
        self.assertIn("empty", str(ctx.exception))

    def test_invalid_mode_raises(self) -> None:
        """Test that invalid mode raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CycleState(
                cycle=0,
                cycle_seed=42,
                slice_name="test",
                mode="invalid_mode",
                candidate_items=["a"],
            )
        self.assertIn("baseline", str(ctx.exception))
        self.assertIn("rfl", str(ctx.exception))

    def test_empty_candidates_raises(self) -> None:
        """Test that empty candidate_items raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            CycleState(
                cycle=0,
                cycle_seed=42,
                slice_name="test_slice",
                mode="baseline",
                candidate_items=[],
            )
        self.assertIn("empty", str(ctx.exception))
        self.assertIn("test_slice", str(ctx.exception))

    def test_valid_baseline_state(self) -> None:
        """Test that valid baseline state is accepted."""
        state = CycleState(
            cycle=0,
            cycle_seed=42,
            slice_name="test",
            mode="baseline",
            candidate_items=["a", "b"],
        )
        self.assertEqual(state.mode, "baseline")

    def test_valid_rfl_state(self) -> None:
        """Test that valid RFL state is accepted."""
        state = CycleState(
            cycle=10,
            cycle_seed=12345,
            slice_name="rfl_test",
            mode="rfl",
            candidate_items=["x"],
        )
        self.assertEqual(state.mode, "rfl")


class TestRflOrderingStrategyValidation(unittest.TestCase):
    """Tests for RflOrderingStrategy input validation."""

    def test_none_policy_raises(self) -> None:
        """Test that None policy raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            RflOrderingStrategy(None)
        self.assertIn("non-None", str(ctx.exception))

    def test_valid_policy_accepted(self) -> None:
        """Test that valid policy is accepted."""
        class MockPolicy:
            def score(self, items):
                return [1.0] * len(items)

        strategy = RflOrderingStrategy(MockPolicy())
        self.assertIsNotNone(strategy.policy)


class TestGetOrderingStrategy(unittest.TestCase):
    """Tests for get_ordering_strategy factory function."""

    def test_baseline_mode(self) -> None:
        """Test baseline mode returns BaselineOrderingStrategy."""
        strategy = get_ordering_strategy("baseline")
        self.assertIsInstance(strategy, BaselineOrderingStrategy)

    def test_rfl_mode_with_policy(self) -> None:
        """Test RFL mode with policy returns RflOrderingStrategy."""
        class MockPolicy:
            def score(self, items):
                return [1.0] * len(items)

        strategy = get_ordering_strategy("rfl", policy=MockPolicy())
        self.assertIsInstance(strategy, RflOrderingStrategy)

    def test_rfl_mode_without_policy_raises(self) -> None:
        """Test RFL mode without policy raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            get_ordering_strategy("rfl")
        self.assertIn("policy", str(ctx.exception).lower())

    def test_unknown_mode_raises(self) -> None:
        """Test unknown mode raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            get_ordering_strategy("unknown")
        self.assertIn("baseline", str(ctx.exception))
        self.assertIn("rfl", str(ctx.exception))


class TestCycleExecutionError(unittest.TestCase):
    """Tests for CycleExecutionError exception."""

    def test_error_has_context(self) -> None:
        """Test CycleExecutionError stores error context."""
        ctx = ErrorContext(
            kind=RuntimeErrorKind.SUBPROCESS,
            message="Command failed",
            slice_name="test",
            cycle=5,
        )
        error = CycleExecutionError("test error", ctx)
        
        self.assertEqual(error.error_context, ctx)
        self.assertIn("Command failed", str(error))

    def test_error_has_original_exception(self) -> None:
        """Test CycleExecutionError stores original exception."""
        ctx = ErrorContext(
            kind=RuntimeErrorKind.UNKNOWN,
            message="original error",
        )
        original = RuntimeError("the original")
        error = CycleExecutionError("wrapped", ctx, original)
        
        self.assertEqual(error.original_exception, original)


class TestExecuteCycleRaiseOnError(unittest.TestCase):
    """Tests for execute_cycle raise_on_error flag."""

    def _failing_substrate(self, item: str, seed: int) -> Dict[str, Any]:
        """Substrate that always fails."""
        raise RuntimeError("Simulated failure")

    def test_raise_on_error_false_returns_result(self) -> None:
        """Test that raise_on_error=False returns failed result."""
        state = CycleState(
            cycle=0,
            cycle_seed=42,
            slice_name="test",
            mode="baseline",
            candidate_items=["a"],
        )
        strategy = BaselineOrderingStrategy()
        
        result = execute_cycle(
            state, strategy, self._failing_substrate,
            raise_on_error=False,
        )
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertIsNotNone(result.error_context)

    def test_raise_on_error_true_raises(self) -> None:
        """Test that raise_on_error=True raises CycleExecutionError."""
        state = CycleState(
            cycle=0,
            cycle_seed=42,
            slice_name="test",
            mode="baseline",
            candidate_items=["a"],
        )
        strategy = BaselineOrderingStrategy()
        
        with self.assertRaises(CycleExecutionError) as ctx:
            execute_cycle(
                state, strategy, self._failing_substrate,
                raise_on_error=True,
            )
        
        self.assertIsInstance(ctx.exception.error_context, ErrorContext)


class TestCycleResultErrorContext(unittest.TestCase):
    """Tests for CycleResult error_context field."""

    def test_successful_result_has_no_error_context(self) -> None:
        """Test successful result has no error context."""
        def success_substrate(item: str, seed: int) -> Dict[str, Any]:
            return {"outcome": "VERIFIED"}

        state = CycleState(
            cycle=0,
            cycle_seed=42,
            slice_name="test",
            mode="baseline",
            candidate_items=["a"],
        )
        strategy = BaselineOrderingStrategy()
        
        result = execute_cycle(state, strategy, success_substrate)
        
        self.assertTrue(result.success)
        self.assertIsNone(result.error_context)

    def test_failed_result_has_error_context(self) -> None:
        """Test failed result has error context with slice/cycle."""
        def failing_substrate(item: str, seed: int) -> Dict[str, Any]:
            raise FileNotFoundError("missing.txt")

        state = CycleState(
            cycle=7,
            cycle_seed=42,
            slice_name="file_slice",
            mode="rfl",
            candidate_items=["a"],
        )
        
        class MockPolicy:
            def score(self, items):
                return [1.0] * len(items)
        
        strategy = RflOrderingStrategy(MockPolicy())
        
        result = execute_cycle(state, strategy, failing_substrate)
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_context)
        self.assertEqual(result.error_context.slice_name, "file_slice")
        self.assertEqual(result.error_context.cycle, 7)
        self.assertEqual(result.error_context.mode, "rfl")


class TestStrategyNameProperty(unittest.TestCase):
    """Tests for strategy name property."""

    def test_baseline_strategy_name(self) -> None:
        """Test BaselineOrderingStrategy has name property."""
        strategy = BaselineOrderingStrategy()
        self.assertEqual(strategy.name, "BaselineOrderingStrategy")

    def test_rfl_strategy_name(self) -> None:
        """Test RflOrderingStrategy has name property."""
        class MockPolicy:
            def score(self, items):
                return [1.0] * len(items)

        strategy = RflOrderingStrategy(MockPolicy())
        self.assertEqual(strategy.name, "RflOrderingStrategy")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

