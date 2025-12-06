# PHASE II — NOT USED IN PHASE I
#
# This script runs a U2 uplift experiment. It is designed to be deterministic
# and self-contained for reproducibility. It supports two modes: 'baseline'
# for random ordering and 'rfl' for policy-driven ordering.
#
# Absolute Safeguards:
# - Do NOT reinterpret Phase I logs as uplift evidence.
# - All Phase II artifacts must be clearly labeled “PHASE II — NOT USED IN PHASE I”.
# - All code must remain deterministic except random shuffle in the baseline policy.
# - RFL uses verifiable feedback only (no RLHF, no preferences, no proxy rewards).
# - All new files must be standalone and MUST NOT modify Phase I behavior.

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from experiments.u2_safe_eval import safe_eval_arithmetic


@dataclass
class TelemetryRecord:
    """
    Represents a single cycle's telemetry in a Phase II U2 experiment.

    This dataclass captures all relevant information for a single experiment
    cycle, including the selected item, execution result, and success status.

    Attributes:
        cycle: The zero-indexed cycle number.
        slice: The experiment slice name (e.g., 'arithmetic_simple').
        mode: The execution mode ('baseline' or 'rfl').
        seed: The random seed used for this cycle.
        item: The problem item that was selected and executed.
        result: String representation of the execution result.
        success: Whether the execution was successful per the slice metric.
        label: Phase label for artifact tracking.
    """

    cycle: int
    slice: str
    mode: str
    seed: int
    item: str
    result: str
    success: bool
    label: str = "PHASE II — NOT USED IN PHASE I"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            A dict with all fields, preserving JSON key names.
        """
        return asdict(self)

# --- Slice-specific Success Metrics ---
# These are passed as pure functions. In a real scenario, these might be
# dynamically imported or otherwise more complex. For this standalone script,
# we define them here.


def metric_arithmetic_simple(item: str, result: Any) -> bool:
    """
    Check if arithmetic expression evaluation matches the expected result.

    Uses safe AST-based evaluation instead of eval() for security.

    Args:
        item: An arithmetic expression string (e.g., "1 + 2 * 3").
        result: The expected numeric result.

    Returns:
        True if safe evaluation of item equals result, False otherwise.
    """
    try:
        evaluated = safe_eval_arithmetic(item)
        if evaluated is None:
            return False
        return evaluated == result
    except (ValueError, SyntaxError):
        return False


def metric_algebra_expansion(item: str, result: Any) -> bool:
    """
    A mock success metric for algebra based on string length expansion.

    This is a placeholder metric. A real implementation would perform
    proper algebraic expansion verification.

    Args:
        item: The original algebra expression.
        result: The expanded result.

    Returns:
        True if the result is longer than the original expression.
    """
    return len(str(result)) > len(item)

METRIC_DISPATCHER: Dict[str, Callable[[str, Any], bool]] = {
    "arithmetic_simple": metric_arithmetic_simple,
    "algebra_expansion": metric_algebra_expansion,
}


# --- RFL Policy Stubs ---
# Mock implementation of the RFL policy scoring and update loop.


class RFLPolicy:
    """
    A mock RFL (Reflexive Formal Learning) policy model.

    This is a simplified policy that scores items based on past success
    feedback. It uses a simple multiplicative update rule.

    Attributes:
        scores: Dictionary mapping items to their current scores.
        rng: Random number generator for initializing new item scores.
    """

    def __init__(self, seed: int) -> None:
        """
        Initialize the RFL policy.

        Args:
            seed: Random seed for deterministic score initialization.
        """
        self.scores: Dict[str, float] = {}
        self.rng = random.Random(seed)

    def score(self, items: List[str]) -> List[float]:
        """
        Score a list of items. Higher scores indicate higher priority.

        Items not seen before are assigned a random initial score.

        Args:
            items: List of item strings to score.

        Returns:
            List of float scores corresponding to each input item.
        """
        for item in items:
            if item not in self.scores:
                self.scores[item] = self.rng.random()
        return [self.scores[item] for item in items]

    def update(self, item: str, success: bool) -> None:
        """
        Update the policy based on execution feedback.

        Uses a simple multiplicative update rule:
        - Success: multiply score by 1.1
        - Failure: multiply score by 0.9
        Scores are clamped to [0.01, 0.99].

        Args:
            item: The item that was executed.
            success: Whether the execution was successful.
        """
        if success:
            self.scores[item] = self.scores.get(item, 0.5) * 1.1
        else:
            self.scores[item] = self.scores.get(item, 0.5) * 0.9
        # Clamp scores to a reasonable range
        self.scores[item] = max(0.01, min(self.scores[item], 0.99))


# --- Core Runner Logic ---


def get_config(config_path: Path) -> Dict[str, Any]:
    """
    Load the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The parsed configuration dictionary.

    Side-effects:
        Prints info message to stdout.
        Exits with code 1 if file not found.
    """
    print(f"INFO: Loading config from {config_path}")
    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_seed_schedule(initial_seed: int, num_cycles: int) -> List[int]:
    """
    Generate a deterministic list of seeds for each cycle.

    Args:
        initial_seed: The master seed for reproducibility.
        num_cycles: Number of experiment cycles.

    Returns:
        A list of integer seeds, one per cycle.
    """
    rng = random.Random(initial_seed)
    return [rng.randint(0, 2**32 - 1) for _ in range(num_cycles)]


def hash_string(data: str) -> str:
    """
    Compute the SHA256 hash of a string.

    Args:
        data: The string to hash.

    Returns:
        Hexadecimal SHA256 digest.
    """
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def _execute_item(chosen_item: str, slice_name: str) -> Any:
    """
    Execute a chosen item and return the result.

    Uses safe AST-based evaluation for arithmetic slices.

    Args:
        chosen_item: The problem item to execute.
        slice_name: The slice determining execution strategy.

    Returns:
        The execution result (numeric for arithmetic, string for others).
    """
    if slice_name == "arithmetic_simple":
        result = safe_eval_arithmetic(chosen_item)
        # Return the result or None if evaluation failed
        return result
    else:
        return f"Expanded({chosen_item})"


def _select_item_baseline(items: List[str], rng: random.Random) -> str:
    """
    Select an item using baseline random shuffle strategy.

    Args:
        items: List of candidate items.
        rng: Random number generator for this cycle.

    Returns:
        The selected item (first after shuffle).
    """
    ordered_items = list(items)
    rng.shuffle(ordered_items)
    return ordered_items[0]


def _select_item_rfl(items: List[str], policy: "RFLPolicy") -> str:
    """
    Select an item using RFL policy scoring.

    Args:
        items: List of candidate items.
        policy: The RFL policy model.

    Returns:
        The highest-scoring item per policy.
    """
    item_scores = policy.score(items)
    scored_items = sorted(zip(items, item_scores), key=lambda x: x[1], reverse=True)
    return scored_items[0][0]


def _run_single_cycle(
    cycle_idx: int,
    cycle_seed: int,
    items: List[str],
    slice_name: str,
    mode: str,
    policy: Optional["RFLPolicy"],
    success_metric: Callable[[str, Any], bool],
) -> TelemetryRecord:
    """
    Execute a single experiment cycle.

    Args:
        cycle_idx: Zero-indexed cycle number.
        cycle_seed: Random seed for this cycle.
        items: List of candidate items.
        slice_name: The experiment slice name.
        mode: Execution mode ('baseline' or 'rfl').
        policy: RFL policy (None for baseline mode).
        success_metric: Function to evaluate success.

    Returns:
        TelemetryRecord for this cycle.

    Side-effects:
        Updates RFL policy if in 'rfl' mode.
    """
    rng = random.Random(cycle_seed)

    # --- Ordering Step ---
    if mode == "baseline":
        chosen_item = _select_item_baseline(items, rng)
    elif mode == "rfl":
        chosen_item = _select_item_rfl(items, policy)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # --- Execution & Evaluation ---
    mock_result = _execute_item(chosen_item, slice_name)
    success = success_metric(chosen_item, mock_result)

    # --- RFL Policy Update ---
    if mode == "rfl" and policy is not None:
        policy.update(chosen_item, success)

    return TelemetryRecord(
        cycle=cycle_idx,
        slice=slice_name,
        mode=mode,
        seed=cycle_seed,
        item=chosen_item,
        result=str(mock_result),
        success=success,
    )


def run_experiment(
    slice_name: str,
    cycles: int,
    seed: int,
    mode: str,
    out_dir: Path,
    config: Dict[str, Any],
) -> None:
    """
    Run a Phase II U2 uplift experiment.

    This is the main orchestrator function that sets up the experiment,
    runs all cycles, and writes results and manifest files.

    Args:
        slice_name: The experiment slice to run (e.g., 'arithmetic_simple').
        cycles: Number of experiment cycles.
        seed: Initial random seed for deterministic execution.
        mode: Execution mode ('baseline' or 'rfl').
        out_dir: Output directory for results and manifest files.
        config: Parsed YAML configuration dictionary.

    Side-effects:
        Creates output directory if needed.
        Writes results JSONL file.
        Writes manifest JSON file.
        Prints progress to stdout.
        Exits with code 1 on configuration errors.

    Invariants:
        Assumes config has been loaded via get_config().
        Assumes slice_name exists in config['slices'].
    """
    print(f"--- Running Experiment: slice={slice_name}, mode={mode}, cycles={cycles}, seed={seed} ---")
    print(f"PHASE II — NOT USED IN PHASE I")

    # 1. Setup
    out_dir.mkdir(exist_ok=True)
    slice_config = config.get("slices", {}).get(slice_name)
    if not slice_config:
        print(f"ERROR: Slice '{slice_name}' not found in config.", file=sys.stderr)
        sys.exit(1)

    items = slice_config["items"]
    success_metric = METRIC_DISPATCHER.get(slice_name)
    if not success_metric:
        print(f"ERROR: Success metric for slice '{slice_name}' not found.", file=sys.stderr)
        sys.exit(1)

    seed_schedule = generate_seed_schedule(seed, cycles)
    policy = RFLPolicy(seed) if mode == "rfl" else None
    ht_series: List[Dict[str, Any]] = []

    results_path = out_dir / f"uplift_u2_{slice_name}_{mode}.jsonl"
    manifest_path = out_dir / f"uplift_u2_manifest_{slice_name}_{mode}.json"

    # 2. Main Loop
    with open(results_path, "w") as results_f:
        for i in range(cycles):
            telemetry = _run_single_cycle(
                cycle_idx=i,
                cycle_seed=seed_schedule[i],
                items=items,
                slice_name=slice_name,
                mode=mode,
                policy=policy,
                success_metric=success_metric,
            )
            # Convert to dict for serialization (preserves JSON keys)
            record_dict = telemetry.to_dict()
            ht_series.append(record_dict)
            results_f.write(json.dumps(record_dict) + "\n")
            print(f"Cycle {i+1}/{cycles}: Chose '{telemetry.item}', Success: {telemetry.success}")

    # 3. Manifest Generation
    slice_config_str = json.dumps(slice_config, sort_keys=True)
    slice_config_hash = hash_string(slice_config_str)
    ht_series_str = json.dumps(ht_series, sort_keys=True)
    ht_series_hash = hash_string(ht_series_str)

    manifest = {
        "label": "PHASE II — NOT USED IN PHASE I",
        "slice": slice_name,
        "mode": mode,
        "cycles": cycles,
        "initial_seed": seed,
        "slice_config_hash": slice_config_hash,
        "prereg_hash": slice_config.get("prereg_hash", "N/A"),
        "ht_series_hash": ht_series_hash,
        "deterministic_seed_schedule": seed_schedule,
        "outputs": {
            "results": str(results_path),
            "manifest": str(manifest_path),
        }
    }

    with open(manifest_path, "w") as manifest_f:
        json.dump(manifest, manifest_f, indent=2)

    print(f"\n--- Experiment Complete ---")
    print(f"Results written to {results_path}")
    print(f"Manifest written to {manifest_path}")

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PHASE II U2 Uplift Runner. Must not be used for Phase I.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Absolute Safeguards:
- Do NOT reinterpret Phase I logs as uplift evidence.
- All Phase II artifacts must be clearly labeled.
- RFL uses verifiable feedback only.
        """
    )
    parser.add_argument("--slice", required=True, type=str, help="The experiment slice to run (e.g., 'arithmetic_simple').")
    parser.add_argument("--cycles", required=True, type=int, help="Number of experiment cycles to run.")
    parser.add_argument("--seed", required=True, type=int, help="Initial random seed for deterministic execution.")
    parser.add_argument("--mode", required=True, choices=["baseline", "rfl"], help="Execution mode: 'baseline' or 'rfl'.")
    parser.add_argument("--out", required=True, type=str, help="Output directory for results and manifest files.")
    parser.add_argument("--config", default="config/curriculum_uplift_phase2.yaml", type=str, help="Path to the curriculum config file.")

    args = parser.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out)

    config = get_config(config_path)

    run_experiment(
        slice_name=args.slice,
        cycles=args.cycles,
        seed=args.seed,
        mode=args.mode,
        out_dir=out_dir,
        config=config,
    )

if __name__ == "__main__":
    main()
