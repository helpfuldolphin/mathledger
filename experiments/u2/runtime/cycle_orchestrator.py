"""
PHASE II — NOT USED IN PHASE I

Cycle Orchestrator Module
=========================

Provides per-cycle execution orchestration for U2 uplift experiments.
This module is the **single source of truth** for cycle execution logic.

All Phase II experiment runners should use this module for per-cycle
execution rather than implementing ad-hoc loops. This ensures:
    - Consistent ordering behavior across all experiments
    - Centralized error handling via error_classifier
    - Uniform telemetry emission via trace_logger

This module defines:
    - CycleState: Immutable state for a single experiment cycle
    - CycleResult: Outcome of cycle execution
    - OrderingStrategy: Protocol for item ordering strategies
    - BaselineOrderingStrategy: Random shuffle ordering
    - RflOrderingStrategy: Policy-driven ordering
    - execute_cycle: The canonical cycle execution function
    - CycleExecutionError: Exception with structured context

Example:
    >>> state = CycleState(
    ...     cycle=0,
    ...     cycle_seed=123456,
    ...     slice_name="arithmetic_simple",
    ...     mode="baseline",
    ...     candidate_items=["1+1", "2+2", "3+3"]
    ... )
    >>> strategy = BaselineOrderingStrategy()
    >>> result = execute_cycle(state, strategy, mock_substrate)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from experiments.u2.runtime.error_classifier import (
    classify_error_with_context,
    RuntimeErrorKind,
    ErrorContext,
)


class CycleExecutionError(Exception):
    """
    Exception raised when cycle execution fails with structured context.
    
    This exception wraps the underlying error with experiment context,
    enabling actionable error messages.
    
    Attributes:
        error_context: Structured error information.
        original_exception: The underlying exception that caused the failure.
    """
    
    def __init__(
        self,
        message: str,
        error_context: ErrorContext,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.error_context = error_context
        self.original_exception = original_exception
    
    def __str__(self) -> str:
        return self.error_context.format_message()


@dataclass(frozen=True)
class CycleState:
    """
    Immutable state container for a single experiment cycle.

    This dataclass captures all inputs needed to execute one cycle of the
    U2 uplift experiment, enabling deterministic replay and testing.

    Attributes:
        cycle: Zero-based cycle index within the experiment.
        cycle_seed: Deterministic seed for this cycle's RNG.
        slice_name: Name of the curriculum slice being evaluated.
        mode: Execution mode - "baseline" or "rfl".
        candidate_items: List of items to choose from for this cycle.

    Raises:
        ValueError: If cycle < 0, mode is invalid, or candidate_items is empty.

    Example:
        >>> state = CycleState(
        ...     cycle=0,
        ...     cycle_seed=42,
        ...     slice_name="test_slice",
        ...     mode="baseline",
        ...     candidate_items=["a", "b", "c"]
        ... )
    """

    cycle: int
    cycle_seed: int
    slice_name: str
    mode: str
    candidate_items: List[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate state invariants with clear error messages."""
        if self.cycle < 0:
            raise ValueError(
                f"cycle must be non-negative, got {self.cycle}"
            )
        if not self.slice_name:
            raise ValueError("slice_name cannot be empty")
        if self.mode not in ("baseline", "rfl"):
            raise ValueError(
                f"mode must be 'baseline' or 'rfl', got '{self.mode}'"
            )
        if not self.candidate_items:
            raise ValueError(
                f"candidate_items cannot be empty for slice '{self.slice_name}'"
            )


@dataclass(frozen=True)
class CycleResult:
    """
    Immutable result container for a completed experiment cycle.

    Attributes:
        cycle: Zero-based cycle index that produced this result.
        chosen_item: The item selected by the ordering strategy.
        success: Whether the substrate execution succeeded.
        metric_value: Numeric metric value from evaluation (0.0 if failed).
        raw_result: Raw output dictionary from substrate execution.
        error_message: Error description if execution failed, None otherwise.
        error_context: Structured error context if execution failed.

    Example:
        >>> result = CycleResult(
        ...     cycle=0,
        ...     chosen_item="1+1",
        ...     success=True,
        ...     metric_value=1.0,
        ...     raw_result={"outcome": "VERIFIED"}
        ... )
    """

    cycle: int
    chosen_item: Any
    success: bool
    metric_value: float = 0.0
    raw_result: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    error_context: Optional[ErrorContext] = None


@runtime_checkable
class OrderingStrategy(Protocol):
    """
    Protocol defining the interface for item ordering strategies.

    Implementations must provide an `order` method that takes a list of items
    and a seeded RNG, returning the items in the desired order.

    The first item in the returned list is the "chosen" item for the cycle.
    """

    def order(self, items: List[Any], rng: random.Random) -> List[Any]:
        """
        Order items according to the strategy.

        Args:
            items: List of candidate items to order.
            rng: Seeded random number generator for deterministic ordering.

        Returns:
            Items in strategy-determined order. First item is the chosen one.
        """
        ...


class BaselineOrderingStrategy:
    """
    Baseline ordering strategy using seeded random shuffle.

    This strategy implements the baseline condition for U2 experiments,
    where items are randomly shuffled using the cycle's deterministic seed.

    The implementation matches the original inline behavior:
    ```python
    ordered_items = list(items)
    rng.shuffle(ordered_items)
    chosen_item = ordered_items[0]
    ```

    Example:
        >>> strategy = BaselineOrderingStrategy()
        >>> rng = random.Random(42)
        >>> items = ["a", "b", "c", "d"]
        >>> ordered = strategy.order(items, rng)
        >>> # Result is deterministic given the seed
    """

    def order(self, items: List[Any], rng: random.Random) -> List[Any]:
        """
        Shuffle items randomly using the provided RNG.

        Args:
            items: List of candidate items to shuffle.
            rng: Seeded random number generator.

        Returns:
            Copy of items in shuffled order.
        """
        ordered = list(items)
        rng.shuffle(ordered)
        return ordered
    
    @property
    def name(self) -> str:
        """Return strategy name for introspection."""
        return "BaselineOrderingStrategy"


class RflOrderingStrategy:
    """
    RFL (Reflexive Formal Learning) ordering strategy using policy scoring.

    This strategy implements the RFL condition for U2 experiments,
    where items are ordered by their policy scores (highest first).

    The implementation matches the original inline behavior:
    ```python
    item_scores = policy.score(items)
    scored_items = sorted(zip(items, item_scores), key=lambda x: x[1], reverse=True)
    chosen_item = scored_items[0][0]
    ```

    Attributes:
        policy: An object with a `score(items) -> List[float]` method.

    Example:
        >>> class MockPolicy:
        ...     def score(self, items):
        ...         return [len(str(item)) for item in items]
        >>> strategy = RflOrderingStrategy(MockPolicy())
        >>> rng = random.Random(42)  # Not used by RFL ordering
        >>> ordered = strategy.order(["a", "bb", "ccc"], rng)
        >>> ordered[0]  # Highest scoring item
        'ccc'
    """

    def __init__(self, policy: Any) -> None:
        """
        Initialize with a policy object.

        Args:
            policy: Object with `score(items: List) -> List[float]` method.
        
        Raises:
            ValueError: If policy is None.
        """
        if policy is None:
            raise ValueError("RflOrderingStrategy requires a non-None policy")
        self.policy = policy

    def order(self, items: List[Any], rng: random.Random) -> List[Any]:
        """
        Order items by policy score, highest first.

        Args:
            items: List of candidate items to score and order.
            rng: Seeded RNG (not used by RFL, but required by protocol).

        Returns:
            Items ordered by descending policy score.
        """
        # RNG is not used for RFL ordering, but accepted for protocol compliance
        _ = rng

        scores = self.policy.score(items)
        scored_items = sorted(
            zip(items, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [item for item, _ in scored_items]
    
    @property
    def name(self) -> str:
        """Return strategy name for introspection."""
        return "RflOrderingStrategy"


# Type alias for substrate execution function
SubstrateExecutor = Callable[[str, int], Dict[str, Any]]

# Type alias for pre/post cycle hooks
CycleHook = Callable[[CycleState, Optional[CycleResult]], None]


def execute_cycle(
    state: CycleState,
    strategy: OrderingStrategy,
    substrate: SubstrateExecutor,
    *,
    pre_hook: Optional[CycleHook] = None,
    post_hook: Optional[CycleHook] = None,
    raise_on_error: bool = False,
) -> CycleResult:
    """
    Execute a single experiment cycle.

    This function orchestrates one iteration of the U2 uplift experiment:
    1. Call pre_hook if provided
    2. Order candidate items using the provided strategy
    3. Select the first (highest priority) item
    4. Execute the substrate with the chosen item
    5. Call post_hook if provided
    6. Return the structured result

    This is the **single source of truth** for cycle execution. All Phase II
    experiment runners should use this function rather than implementing
    ad-hoc cycle loops.

    Args:
        state: Immutable state for this cycle.
        strategy: Ordering strategy (baseline or RFL).
        substrate: Callable that executes the chosen item.
            Signature: (item: str, seed: int) -> Dict[str, Any]
            Should return {"outcome": "VERIFIED"} on success.
        pre_hook: Optional callback invoked before execution.
        post_hook: Optional callback invoked after execution.
        raise_on_error: If True, raise CycleExecutionError on substrate failure.

    Returns:
        CycleResult containing execution outcome.

    Raises:
        CycleExecutionError: If raise_on_error=True and substrate fails.

    Example:
        >>> def mock_substrate(item: str, seed: int) -> Dict[str, Any]:
        ...     return {"outcome": "VERIFIED", "value": 42}
        >>> state = CycleState(
        ...     cycle=0, cycle_seed=42, slice_name="test",
        ...     mode="baseline", candidate_items=["a", "b"]
        ... )
        >>> strategy = BaselineOrderingStrategy()
        >>> result = execute_cycle(state, strategy, mock_substrate)
        >>> result.success
        True
    """
    # Create cycle-specific RNG
    rng = random.Random(state.cycle_seed)

    # Order items using strategy
    ordered_items = strategy.order(list(state.candidate_items), rng)
    chosen_item = ordered_items[0]

    # Call pre-hook if provided
    if pre_hook is not None:
        pre_hook(state, None)

    # Execute substrate with error classification
    result: CycleResult
    try:
        raw_result = substrate(chosen_item, state.cycle_seed)
        success = raw_result.get("outcome") == "VERIFIED"
        metric_value = float(raw_result.get("metric_value", 1.0 if success else 0.0))

        result = CycleResult(
            cycle=state.cycle,
            chosen_item=chosen_item,
            success=success,
            metric_value=metric_value,
            raw_result=raw_result,
            error_message=None,
            error_context=None,
        )

    except Exception as e:
        # Classify error with experiment context
        error_ctx = classify_error_with_context(
            e,
            slice_name=state.slice_name,
            cycle=state.cycle,
            mode=state.mode,
            seed=state.cycle_seed,
        )

        result = CycleResult(
            cycle=state.cycle,
            chosen_item=chosen_item,
            success=False,
            metric_value=0.0,
            raw_result={"error": error_ctx.message, "error_kind": error_ctx.kind.value},
            error_message=error_ctx.format_message(),
            error_context=error_ctx,
        )

        if raise_on_error:
            raise CycleExecutionError(
                error_ctx.format_message(),
                error_ctx,
                e,
            ) from e

    # Call post-hook if provided
    if post_hook is not None:
        post_hook(state, result)

    return result


def get_ordering_strategy(
    mode: str,
    policy: Optional[Any] = None,
) -> OrderingStrategy:
    """
    Get the appropriate ordering strategy for a mode.

    This is a convenience factory function for obtaining the correct
    strategy based on experiment mode.

    Args:
        mode: Execution mode ("baseline" or "rfl").
        policy: Policy object (required for "rfl" mode).

    Returns:
        OrderingStrategy instance.

    Raises:
        ValueError: If mode is invalid or policy is missing for RFL.
    """
    if mode == "baseline":
        return BaselineOrderingStrategy()
    elif mode == "rfl":
        if policy is None:
            raise ValueError("RFL mode requires a policy object")
        return RflOrderingStrategy(policy)
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Expected 'baseline' or 'rfl'.")


__all__ = [
    "CycleState",
    "CycleResult",
    "CycleExecutionError",
    "OrderingStrategy",
    "BaselineOrderingStrategy",
    "RflOrderingStrategy",
    "SubstrateExecutor",
    "CycleHook",
    "execute_cycle",
    "get_ordering_strategy",
]
