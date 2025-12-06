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
from pathlib import Path
from typing import Any, Callable, Dict, List

import yaml

# --- Slice-specific Success Metrics ---
# These are passed as pure functions. In a real scenario, these might be
# dynamically imported or otherwise more complex. For this standalone script,
# we define them here.


def metric_arithmetic_simple(item: str, result: Any) -> bool:
    """
    Compute success metric for arithmetic_simple slice.

    Args:
        item: The input expression string (e.g., "2+3").
        result: The computed result to verify.

    Returns:
        True if the Python eval of the item matches the result, False otherwise.
    """
    try:
        return eval(item) == result
    except Exception:
        return False


def metric_algebra_expansion(item: str, result: Any) -> bool:
    """
    Compute success metric for algebra_expansion slice.

    This is a placeholder metric. A real implementation would validate
    algebraic expansion correctness.

    Args:
        item: The input expression string.
        result: The computed result to verify.

    Returns:
        True if the result string is longer than the input (indicating expansion).
    """
    return len(str(result)) > len(item)


def get_success_metric(slice_name: str) -> Callable[[str, Any], bool]:
    """
    Retrieve the success metric function for a given slice name.

    PHASE II — NOT USED IN PHASE I

    Args:
        slice_name: The name of the experiment slice (e.g., "arithmetic_simple").

    Returns:
        A callable that takes (item, result) and returns True if the result
        is considered successful for the given slice.

    Raises:
        KeyError: If no metric is registered for the given slice name.
    """
    metric_dispatcher: Dict[str, Callable[[str, Any], bool]] = {
        "arithmetic_simple": metric_arithmetic_simple,
        "algebra_expansion": metric_algebra_expansion,
    }
    if slice_name not in metric_dispatcher:
        raise KeyError(f"No success metric registered for slice '{slice_name}'")
    return metric_dispatcher[slice_name]


# --- RFL Policy Stubs ---
# Mock implementation of the RFL policy scoring and update loop.


class RFLPolicy:
    """
    A mock RFL policy model for uplift experiments.

    PHASE II — NOT USED IN PHASE I

    This class implements a simple scoring and update mechanism for policy-driven
    ordering of experiment items. It maintains internal scores for each item and
    updates them based on success/failure feedback.

    Attributes:
        scores: Dictionary mapping item strings to their current scores.
        rng: Random number generator for deterministic score initialization.
    """

    def __init__(self, seed: int):
        """
        Initialize the RFL policy with a deterministic seed.

        Args:
            seed: Random seed for deterministic score initialization.
        """
        self.scores: Dict[str, float] = {}
        self.rng = random.Random(seed)

    def score(self, items: List[str]) -> List[float]:
        """
        Score a list of items for ordering.

        Items not previously seen are assigned random initial scores.
        Higher scores indicate higher priority.

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
        Update the policy based on feedback from an experiment cycle.

        Args:
            item: The item that was evaluated.
            success: True if the evaluation was successful, False otherwise.
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
    Load the YAML configuration file for the experiment.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the parsed configuration.

    Raises:
        SystemExit: If the configuration file does not exist.
    """
    print(f"INFO: Loading config from {config_path}")
    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_seed_schedule(initial_seed: int, num_cycles: int) -> List[int]:
    """
    Generate a deterministic list of seeds for each experiment cycle.

    PHASE II — NOT USED IN PHASE I

    Args:
        initial_seed: The initial seed for the random number generator.
        num_cycles: The number of cycles (and seeds) to generate.

    Returns:
        A list of integer seeds, one for each cycle.
    """
    rng = random.Random(initial_seed)
    return [rng.randint(0, 2**32 - 1) for _ in range(num_cycles)]


def hash_string(data: str) -> str:
    """
    Compute the SHA256 hash of a string.

    Args:
        data: The input string to hash.

    Returns:
        The hexadecimal SHA256 hash of the input string.
    """
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def build_uplift_manifest(
    slice_name: str,
    slice_config: Dict[str, Any],
    mode: str,
    cycles: int,
    initial_seed: int,
    seed_schedule: List[int],
    ht_series: List[Dict[str, Any]],
    results_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    """
    Build the manifest dictionary for an uplift experiment.

    PHASE II — NOT USED IN PHASE I

    This function constructs a cryptographically bound manifest containing
    configuration hashes and telemetry hashes for reproducibility verification.

    Args:
        slice_name: The name of the experiment slice.
        slice_config: The configuration dictionary for the slice.
        mode: The execution mode ('baseline' or 'rfl').
        cycles: The number of experiment cycles.
        initial_seed: The initial random seed.
        seed_schedule: The list of per-cycle seeds.
        ht_series: The telemetry history series (list of telemetry records).
        results_path: Path to the results JSONL file.
        manifest_path: Path to the manifest JSON file.

    Returns:
        A dictionary containing the manifest data ready for JSON serialization.
    """
    slice_config_str = json.dumps(slice_config, sort_keys=True)
    slice_config_hash = hash_string(slice_config_str)
    ht_series_str = json.dumps(ht_series, sort_keys=True)
    ht_series_hash = hash_string(ht_series_str)

    return {
        "label": "PHASE II — NOT USED IN PHASE I",
        "slice": slice_name,
        "mode": mode,
        "cycles": cycles,
        "initial_seed": initial_seed,
        "slice_config_hash": slice_config_hash,
        "prereg_hash": slice_config.get("prereg_hash", "N/A"),
        "ht_series_hash": ht_series_hash,
        "deterministic_seed_schedule": seed_schedule,
        "outputs": {
            "results": str(results_path),
            "manifest": str(manifest_path),
        }
    }


def write_manifest(manifest: Dict[str, Any], manifest_path: Path) -> None:
    """
    Write a manifest dictionary to a JSON file.

    Args:
        manifest: The manifest dictionary to write.
        manifest_path: The path to write the manifest file to.
    """
    with open(manifest_path, "w") as manifest_f:
        json.dump(manifest, manifest_f, indent=2)


def run_experiment(
    slice_name: str,
    cycles: int,
    seed: int,
    mode: str,
    out_dir: Path,
    config: Dict[str, Any],
) -> None:
    """
    Run a U2 uplift experiment.

    PHASE II — NOT USED IN PHASE I

    This function executes the main experiment loop, running the specified number
    of cycles with either baseline (random) or RFL (policy-driven) ordering.

    Args:
        slice_name: The name of the experiment slice (e.g., "arithmetic_simple").
        cycles: The number of experiment cycles to run.
        seed: The initial random seed for deterministic execution.
        mode: The execution mode ('baseline' or 'rfl').
        out_dir: The output directory for results and manifest files.
        config: The loaded configuration dictionary.

    Raises:
        SystemExit: If the slice or success metric is not found.
        ValueError: If an unknown mode is specified.
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
    try:
        success_metric = get_success_metric(slice_name)
    except KeyError:
        print(f"ERROR: Success metric for slice '{slice_name}' not found.", file=sys.stderr)
        sys.exit(1)

    seed_schedule = generate_seed_schedule(seed, cycles)
    policy = RFLPolicy(seed) if mode == "rfl" else None
    ht_series: List[Dict[str, Any]] = []  # History of Telemetry (Hₜ)

    results_path = out_dir / f"uplift_u2_{slice_name}_{mode}.jsonl"
    manifest_path = out_dir / f"uplift_u2_manifest_{slice_name}_{mode}.json"

    # 2. Main Loop
    with open(results_path, "w") as results_f:
        for cycle in range(cycles):
            cycle_seed = seed_schedule[cycle]
            rng = random.Random(cycle_seed)
            
            # --- Ordering Step ---
            if mode == "baseline":
                # Baseline: random shuffle ordering
                ordered_items = list(items)
                rng.shuffle(ordered_items)
                chosen_item = ordered_items[0]
            elif mode == "rfl":
                # RFL: use policy scoring
                item_scores = policy.score(items)
                scored_items = sorted(zip(items, item_scores), key=lambda x: x[1], reverse=True)
                chosen_item = scored_items[0][0]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # --- Mock Execution & Evaluation ---
            # In a real system, this would be a call to the substrate.
            # Here, we just mock a result.
            mock_result = eval(chosen_item) if slice_name == "arithmetic_simple" else f"Expanded({chosen_item})"
            success = success_metric(chosen_item, mock_result)

            # --- RFL Policy Update ---
            if mode == "rfl":
                policy.update(chosen_item, success)

            # --- Telemetry Logging ---
            telemetry_record = {
                "cycle": cycle,
                "slice": slice_name,
                "mode": mode,
                "seed": cycle_seed,
                "item": chosen_item,
                "result": str(mock_result),
                "success": success,
                "label": "PHASE II — NOT USED IN PHASE I",
            }
            ht_series.append(telemetry_record)
            results_f.write(json.dumps(telemetry_record) + "\n")
            print(f"Cycle {cycle+1}/{cycles}: Chose '{chosen_item}', Success: {success}")

    # 3. Manifest Generation
    manifest = build_uplift_manifest(
        slice_name=slice_name,
        slice_config=slice_config,
        mode=mode,
        cycles=cycles,
        initial_seed=seed,
        seed_schedule=seed_schedule,
        ht_series=ht_series,
        results_path=results_path,
        manifest_path=manifest_path,
    )
    write_manifest(manifest, manifest_path)

    print(f"\n--- Experiment Complete ---")
    print(f"Results written to {results_path}")
    print(f"Manifest written to {manifest_path}")

def main() -> None:
    """
    CLI entry point for the U2 uplift experiment runner.

    PHASE II — NOT USED IN PHASE I

    This function parses command-line arguments and executes the uplift experiment
    with the specified configuration. It supports two modes:
    - 'baseline': Random ordering of items per cycle
    - 'rfl': Policy-driven ordering with feedback-based updates
    """
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
