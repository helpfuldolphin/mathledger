"""
U2 Entry Point - Type-Verified Runner Surface

Provides a clean, typed entry point for U2 experiments with integrated
safety envelope and performance monitoring.
"""

from typing import Callable, List, Tuple, Any, Optional
from pathlib import Path

from .runner import U2Config, U2Runner, CycleResult
from .safety_envelope import U2SafetyEnvelope, build_u2_safety_envelope
from .u2_safe_eval import SafeEvalLintResult, batch_lint_expressions


def run_u2_experiment(
    config: U2Config,
    items: List[str],
    execute_fn: Callable[[str, int], Tuple[bool, Any]],
    lint_expressions: Optional[List[str]] = None,
) -> Tuple[List[CycleResult], U2SafetyEnvelope]:
    """
    Run a U2 experiment with type verification and safety monitoring.
    
    This is the primary entry point for external code to run U2 experiments.
    It provides:
    - Fully typed configuration and results
    - Integrated safety envelope
    - Performance monitoring
    - Optional expression linting
    
    Args:
        config: U2Config with experiment parameters (fully typed)
        items: List of candidate items for the experiment
        execute_fn: Execution function (item, seed) -> (success, result)
        lint_expressions: Optional list of expressions to lint for safety
        
    Returns:
        Tuple of (cycle_results, safety_envelope)
        - cycle_results: List[CycleResult] with typed cycle data
        - safety_envelope: U2SafetyEnvelope with safety assessment
        
    Example:
        >>> config = U2Config(
        ...     experiment_id="test_exp",
        ...     slice_name="arithmetic",
        ...     mode="baseline",
        ...     total_cycles=10,
        ...     master_seed=42,
        ... )
        >>> def execute(item: str, seed: int) -> Tuple[bool, Any]:
        ...     # Execute item and return (success, result)
        ...     return True, {"outcome": "VERIFIED"}
        >>> results, envelope = run_u2_experiment(
        ...     config=config,
        ...     items=["1+1", "2+2", "3+3"],
        ...     execute_fn=execute,
        ... )
        >>> print(f"Safety status: {envelope.safety_status}")
        >>> print(f"Completed {len(results)} cycles")
    """
    # Initialize runner
    runner = U2Runner(config)
    
    # Run all cycles
    cycle_results: List[CycleResult] = []
    for _ in range(config.total_cycles):
        result = runner.run_cycle(items, execute_fn)
        cycle_results.append(result)
        
        # Maybe save snapshot
        runner.maybe_save_snapshot()
    
    # Gather performance stats
    perf_stats = {
        "cycle_durations_ms": runner.cycle_durations_ms,
        "max_cycle_duration_ms": max(runner.cycle_durations_ms) if runner.cycle_durations_ms else 0.0,
        "avg_cycle_duration_ms": (
            sum(runner.cycle_durations_ms) / len(runner.cycle_durations_ms)
            if runner.cycle_durations_ms
            else 0.0
        ),
    }
    
    # Lint expressions if provided
    lint_results: List[SafeEvalLintResult] = []
    if lint_expressions:
        lint_results = batch_lint_expressions(lint_expressions)
    
    # Build safety envelope
    safety_envelope = build_u2_safety_envelope(
        run_config=config,
        perf_stats=perf_stats,
        eval_lint_results=lint_results,
    )
    
    return cycle_results, safety_envelope
