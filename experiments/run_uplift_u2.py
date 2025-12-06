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
from typing import Any, Dict, List

from experiments.curriculum_loader_v2 import (
    CurriculumLoaderV2,
    SuccessMetricSpec,
    UpliftSlice,
)


# --- RFL Policy Stubs ---
# Mock implementation of the RFL policy scoring and update loop.

class RFLPolicy:
    """A mock RFL policy model."""
    def __init__(self, seed: int):
        self.scores = {}
        self.rng = random.Random(seed)

    def score(self, items: List[str]) -> List[float]:
        """Scores items. Higher is better."""
        # Initialize scores if not seen before
        for item in items:
            if item not in self.scores:
                self.scores[item] = self.rng.random()
        return [self.scores[item] for item in items]

    def update(self, item: str, success: bool):
        """Updates the policy based on feedback."""
        # Simple update rule: reward success, penalize failure.
        if success:
            self.scores[item] = self.scores.get(item, 0.5) * 1.1
        else:
            self.scores[item] = self.scores.get(item, 0.5) * 0.9
        # Clamp scores to a reasonable range
        self.scores[item] = max(0.01, min(self.scores[item], 0.99))


# --- Core Runner Logic ---

def get_loader(config_path: Path) -> CurriculumLoaderV2:
    """
    Load curriculum using CurriculumLoaderV2.

    Args:
        config_path: Path to curriculum YAML file.

    Returns:
        CurriculumLoaderV2 instance with validated configuration.
    """
    print(f"INFO: Loading config from {config_path}")
    try:
        return CurriculumLoaderV2.from_yaml_path(config_path)
    except FileNotFoundError:
        print(f"ERROR: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

def generate_seed_schedule(initial_seed: int, num_cycles: int) -> List[int]:
    """Generates a deterministic list of seeds for each cycle."""
    rng = random.Random(initial_seed)
    return [rng.randint(0, 2**32 - 1) for _ in range(num_cycles)]

def hash_string(data: str) -> str:
    """Computes the SHA256 hash of a string."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def evaluate_success(
    slice_obj: UpliftSlice,
    chosen_item: str,
    mock_result: Any,
) -> bool:
    """
    Evaluate success using the slice's SuccessMetricSpec.

    This is a mock evaluator that simulates success based on the metric kind.
    In production, this would call the actual metric functions with real
    verified statements.

    Args:
        slice_obj: UpliftSlice with success_metric configuration.
        chosen_item: The item/formula being evaluated.
        mock_result: The mock execution result.

    Returns:
        True if success criteria met, False otherwise.
    """
    metric_spec = slice_obj.success_metric
    kind = metric_spec.kind

    # Mock evaluation based on metric kind
    # In production, these would call slice_success_metrics functions
    if kind == "sparse":
        # For sparse metric, mock success based on item evaluation
        try:
            # Attempt to evaluate arithmetic expressions
            expected = eval(chosen_item)
            return mock_result == expected
        except Exception:
            # For non-arithmetic items, use length-based heuristic
            return len(str(mock_result)) > len(chosen_item)
    elif kind == "goal_hit":
        # Mock: success if mock_result hash is in target_hashes
        # In production, would use compute_goal_hit
        thresholds = metric_spec.thresholds
        min_verified = thresholds.get("min_total_verified", 1)
        return min_verified <= 1  # Mock: always succeed if threshold is 1
    elif kind == "chain_length":
        # Mock: success based on chain length threshold
        thresholds = metric_spec.thresholds
        min_chain = thresholds.get("min_chain_length", 1)
        return min_chain <= 1  # Mock: always succeed if threshold is 1
    elif kind == "multi_goal":
        # Mock: success if required goals are met
        return True  # Mock: optimistically succeed

    # Default fallback
    return False


def run_experiment(
    slice_name: str,
    cycles: int,
    seed: int,
    mode: str,
    out_dir: Path,
    loader: CurriculumLoaderV2,
):
    """Main function to run the uplift experiment."""
    print(f"--- Running Experiment: slice={slice_name}, mode={mode}, cycles={cycles}, seed={seed} ---")
    print(f"PHASE II — NOT USED IN PHASE I")

    # 1. Setup using CurriculumLoaderV2
    out_dir.mkdir(exist_ok=True)

    try:
        slice_obj = loader.get_slice(slice_name)
    except KeyError:
        print(f"ERROR: Slice '{slice_name}' not found in config.", file=sys.stderr)
        sys.exit(1)

    items = list(slice_obj.items)
    metric_spec = slice_obj.success_metric

    print(f"INFO: Slice '{slice_name}' loaded with metric kind '{metric_spec.kind}'")
    print(f"INFO: Slice config hash: {slice_obj.config_hash}")

    seed_schedule = generate_seed_schedule(seed, cycles)
    policy = RFLPolicy(seed) if mode == "rfl" else None
    ht_series = []  # History of Telemetry (Hₜ)

    results_path = out_dir / f"uplift_u2_{slice_name}_{mode}.jsonl"
    manifest_path = out_dir / f"uplift_u2_manifest_{slice_name}_{mode}.json"

    # 2. Main Loop
    with open(results_path, "w") as results_f:
        for i in range(cycles):
            cycle_seed = seed_schedule[i]
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
            try:
                mock_result = eval(chosen_item)
            except Exception:
                mock_result = f"Expanded({chosen_item})"

            # Use SuccessMetricSpec-driven evaluation
            success = evaluate_success(slice_obj, chosen_item, mock_result)

            # --- RFL Policy Update ---
            if mode == "rfl":
                policy.update(chosen_item, success)

            # --- Telemetry Logging ---
            telemetry_record = {
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
            results_f.write(json.dumps(telemetry_record) + "\n")
            print(f"Cycle {i+1}/{cycles}: Chose '{chosen_item}', Success: {success}")

    # 3. Manifest Generation - use slice_obj from loader
    ht_series_str = json.dumps(ht_series, sort_keys=True)
    ht_series_hash = hash_string(ht_series_str)

    manifest = {
        "label": "PHASE II — NOT USED IN PHASE I",
        "slice": slice_name,
        "mode": mode,
        "cycles": cycles,
        "initial_seed": seed,
        "slice_config_hash": slice_obj.config_hash,
        "prereg_hash": slice_obj.prereg_hash or "N/A",
        "success_metric": metric_spec.to_dict(),
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

    # Use CurriculumLoaderV2 instead of raw YAML loading
    loader = get_loader(config_path)

    run_experiment(
        slice_name=args.slice,
        cycles=args.cycles,
        seed=args.seed,
        mode=args.mode,
        out_dir=out_dir,
        loader=loader,
    )

if __name__ == "__main__":
    main()
