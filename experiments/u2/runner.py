"""
PHASE-II — NOT USED IN PHASE I

U2 Uplift Experiment Runner
===========================

This module provides the main experiment runner for U2 uplift experiments.
It orchestrates the execution of baseline and RFL mode experiments with
full determinism and audit trail logging.

**Determinism Notes:**
    - All random operations use seeded RNG instances.
    - Seed schedules are computed deterministically from the initial seed.
    - Policy updates are deterministic given the same feedback sequence.
    - Same configuration always produces the same results.

Absolute Safeguards:
    - Do NOT reinterpret Phase I logs as uplift evidence.
    - All Phase II artifacts must be clearly labeled "PHASE II — NOT USED IN PHASE I".
    - All code must remain deterministic except random shuffle in the baseline policy.
    - RFL uses verifiable feedback only (no RLHF, no preferences, no proxy rewards).
    - All new files must be standalone and MUST NOT modify Phase I behavior.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from experiments.u2.seed import generate_seed_schedule
from experiments.u2.metrics import get_metric_function, MetricFunction
from experiments.u2.manifest import generate_manifest, save_manifest
from experiments.u2.audit import AuditLogger


class RFLPolicy:
    """A mock RFL policy model for candidate scoring.

    This class implements a simple policy that scores items based on
    learned success correlations. Higher scores indicate items more
    likely to succeed.

    Attributes:
        scores: Dictionary mapping items to their current scores.
        rng: Seeded random number generator for initialization.

    Example:
        >>> policy = RFLPolicy(seed=42)
        >>> scores = policy.score(["a", "b", "c"])
        >>> len(scores)
        3
        >>> policy.update("a", success=True)  # Reinforce successful item

    **Determinism Notes:**
        - Initial scores are deterministic given the seed.
        - Updates are deterministic given the feedback sequence.
        - Same seed and feedback always produces same policy state.
    """

    # Score bounds for clamping
    MIN_SCORE: float = 0.01
    MAX_SCORE: float = 0.99

    # Update multipliers
    SUCCESS_MULTIPLIER: float = 1.1
    FAILURE_MULTIPLIER: float = 0.9

    def __init__(self, seed: int) -> None:
        """Initialize the RFL policy.

        Args:
            seed: Random seed for deterministic initialization.
                Must be non-negative.

        Raises:
            ValueError: If seed is negative.
        """
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")

        self.scores: Dict[str, float] = {}
        self.rng = random.Random(seed)

    def score(self, items: List[str]) -> List[float]:
        """Score a list of items.

        Items not seen before are assigned a random initial score
        using the seeded RNG.

        Args:
            items: List of item strings to score.

        Returns:
            List of scores in the same order as items.

        Example:
            >>> policy = RFLPolicy(42)
            >>> scores = policy.score(["a", "b"])
            >>> all(0 <= s <= 1 for s in scores)
            True
        """
        for item in items:
            if item not in self.scores:
                self.scores[item] = self.rng.random()
        return [self.scores[item] for item in items]

    def update(self, item: str, success: bool) -> None:
        """Update the policy based on feedback.

        Success increases the item's score, failure decreases it.
        Scores are clamped to [MIN_SCORE, MAX_SCORE].

        Args:
            item: The item that was evaluated.
            success: Whether the evaluation was successful.

        Example:
            >>> policy = RFLPolicy(42)
            >>> _ = policy.score(["a"])
            >>> old_score = policy.scores["a"]
            >>> policy.update("a", success=True)
            >>> policy.scores["a"] > old_score
            True
        """
        current_score = self.scores.get(item, 0.5)

        if success:
            new_score = current_score * self.SUCCESS_MULTIPLIER
        else:
            new_score = current_score * self.FAILURE_MULTIPLIER

        # Clamp to valid range
        self.scores[item] = max(self.MIN_SCORE, min(new_score, self.MAX_SCORE))

    def get_best_item(self, items: List[str]) -> str:
        """Get the highest-scoring item from a list.

        Args:
            items: List of items to choose from.

        Returns:
            The item with the highest score.

        Raises:
            ValueError: If items is empty.
        """
        if not items:
            raise ValueError("items cannot be empty")

        scores = self.score(items)
        best_idx = max(range(len(items)), key=lambda i: scores[i])
        return items[best_idx]


def _execute_baseline_ordering(
    items: List[str],
    rng: random.Random,
) -> str:
    """Execute baseline (random) ordering and select the first item.

    Args:
        items: List of items to choose from.
        rng: Seeded random number generator.

    Returns:
        The selected item.

    **Determinism Notes:**
        - Uses the provided RNG for shuffling.
        - Same RNG state always produces same ordering.
    """
    ordered_items = list(items)
    rng.shuffle(ordered_items)
    return ordered_items[0]


def _execute_rfl_ordering(
    items: List[str],
    policy: RFLPolicy,
) -> str:
    """Execute RFL (policy-based) ordering and select the best item.

    Args:
        items: List of items to choose from.
        policy: The RFL policy for scoring.

    Returns:
        The highest-scoring item.

    **Determinism Notes:**
        - Policy scores are deterministic.
        - Same policy state always produces same selection.
    """
    return policy.get_best_item(items)


def _mock_execute_item(item: str, slice_name: str) -> Any:
    """Mock execution of an item for evaluation.

    In a real system, this would call the substrate for actual execution.
    Here we provide mock results for testing.

    Args:
        item: The item to execute.
        slice_name: The experiment slice name.

    Returns:
        The mock result of execution.
    """
    if slice_name == "arithmetic_simple":
        from experiments.u2.metrics import _safe_eval_arithmetic
        return _safe_eval_arithmetic(item)
    else:
        return f"Expanded({item})"


def run_experiment(
    slice_name: str,
    cycles: int,
    seed: int,
    mode: str,
    out_dir: Path,
    config: Dict[str, Any],
) -> None:
    """Run a U2 uplift experiment.

    This function orchestrates the complete experiment execution including:
    - Seed schedule generation
    - Cycle execution (baseline or RFL mode)
    - Audit logging
    - Manifest generation

    Args:
        slice_name: The experiment slice to run (e.g., "arithmetic_simple").
        cycles: Number of experiment cycles to run.
        seed: Initial random seed for deterministic execution.
        mode: Execution mode ("baseline" or "rfl").
        out_dir: Output directory for results and manifest files.
        config: Configuration dictionary containing slice definitions.

    Raises:
        ValueError: If slice_name not found in config or invalid mode.
        SystemExit: On critical errors (with error message to stderr).

    Example:
        >>> config = {"slices": {"arithmetic_simple": {"items": ["1+1", "2+2"]}}}
        >>> run_experiment(
        ...     slice_name="arithmetic_simple",
        ...     cycles=5,
        ...     seed=42,
        ...     mode="baseline",
        ...     out_dir=Path("/tmp/test"),
        ...     config=config,
        ... )  # doctest: +SKIP

    **Determinism Notes:**
        - Same inputs always produce the same outputs.
        - Seed schedule is deterministic.
        - Policy updates (RFL mode) are deterministic.
    """
    print(f"--- Running Experiment: slice={slice_name}, mode={mode}, cycles={cycles}, seed={seed} ---")
    print("PHASE II — NOT USED IN PHASE I")

    # Validate inputs
    if mode not in ("baseline", "rfl"):
        print(f"ERROR: Invalid mode '{mode}'. Expected 'baseline' or 'rfl'.", file=sys.stderr)
        sys.exit(1)

    # Setup output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get slice configuration
    slice_config = config.get("slices", {}).get(slice_name)
    if not slice_config:
        print(
            f"ERROR: Slice '{slice_name}' not found in config. "
            f"Available slices: {list(config.get('slices', {}).keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    items = slice_config.get("items", [])
    if not items:
        print(f"ERROR: Slice '{slice_name}' has no items defined.", file=sys.stderr)
        sys.exit(1)

    # Get success metric
    success_metric = get_metric_function(slice_name)
    if not success_metric:
        print(
            f"ERROR: Success metric for slice '{slice_name}' not found. "
            f"Register it using metrics.register_metric().",
            file=sys.stderr,
        )
        sys.exit(1)

    # Generate seed schedule
    seed_schedule = generate_seed_schedule(seed, cycles)

    # Initialize policy (RFL mode only)
    policy: Optional[RFLPolicy] = RFLPolicy(seed) if mode == "rfl" else None

    # Initialize audit logger
    audit_logger = AuditLogger(slice_name, mode)

    # Telemetry series for manifest
    ht_series: List[Dict[str, Any]] = []

    # Output paths
    results_path = out_dir / f"uplift_u2_{slice_name}_{mode}.jsonl"
    manifest_path = out_dir / f"uplift_u2_manifest_{slice_name}_{mode}.json"
    audit_path = out_dir / f"uplift_u2_audit_{slice_name}_{mode}.json"

    # Main execution loop
    with open(results_path, "w", encoding="utf-8") as results_f:
        for i in range(cycles):
            cycle_seed = seed_schedule[i]
            rng = random.Random(cycle_seed)

            # Select item based on mode
            if mode == "baseline":
                chosen_item = _execute_baseline_ordering(items, rng)
                policy_score = None
            else:
                chosen_item = _execute_rfl_ordering(items, policy)
                policy_score = policy.scores.get(chosen_item)

            # Execute and evaluate
            mock_result = _mock_execute_item(chosen_item, slice_name)
            success = success_metric(chosen_item, mock_result)

            # Update RFL policy
            if mode == "rfl" and policy is not None:
                policy.update(chosen_item, success)

            # Create telemetry record
            telemetry_record: Dict[str, Any] = {
                "cycle": i,
                "slice": slice_name,
                "mode": mode,
                "seed": cycle_seed,
                "item": chosen_item,
                "result": str(mock_result),
                "success": success,
                "label": "PHASE II — NOT USED IN PHASE I",
            }
            ht_series.append(telemetry_record)

            # Write to results file
            results_f.write(json.dumps(telemetry_record, sort_keys=True) + "\n")

            # Log to audit trail
            audit_logger.log_cycle(
                cycle=i,
                seed=cycle_seed,
                item=chosen_item,
                result=str(mock_result),
                success=success,
                policy_score=policy_score,
            )

            # Progress output
            if (i + 1) % max(1, cycles // 10) == 0:
                print(f"Cycle {i + 1}/{cycles}: Chose '{chosen_item}', Success: {success}")

    # Generate and save manifest
    manifest = generate_manifest(
        slice_name=slice_name,
        mode=mode,
        cycles=cycles,
        initial_seed=seed,
        slice_config=slice_config,
        prereg_hash=slice_config.get("prereg_hash"),
        ht_series=ht_series,
        seed_schedule=seed_schedule,
        results_path=results_path,
        manifest_path=manifest_path,
    )
    save_manifest(manifest, manifest_path)

    # Export audit log
    audit_logger.export(audit_path)

    # Summary
    print(f"\n--- Experiment Complete ---")
    print(f"Results written to {results_path}")
    print(f"Manifest written to {manifest_path}")
    print(f"Audit log written to {audit_path}")
    print(f"Success rate: {audit_logger.get_success_rate():.2%}")
