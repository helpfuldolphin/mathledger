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

# Add project root to sys.path for module imports
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# --- Slice-specific Success Metrics ---
# These are passed as pure functions. In a real scenario, these might be
# dynamically imported or otherwise more complex. For this standalone script,
# we define them here.

def metric_arithmetic_simple(item: str, result: Any) -> bool:
    """Success is when the python eval matches the expected result."""
    try:
        # A mock 'correct' result is simply the eval of the string.
        return eval(item) == result
    except Exception:
        return False

def metric_algebra_expansion(item: str, result: Any) -> bool:
    """A mock success metric for algebra. We'll just use string length."""
    # This is a placeholder. A real metric would be much more complex.
    return len(str(result)) > len(item)

METRIC_DISPATCHER = {
    "arithmetic_simple": metric_arithmetic_simple,
    "algebra_expansion": metric_algebra_expansion,
}


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

def get_config(config_path: Path) -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    print(f"INFO: Loading config from {config_path}")
    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_seed_schedule(initial_seed: int, num_cycles: int) -> List[int]:
    """Generates a deterministic list of seeds for each cycle."""
    rng = random.Random(initial_seed)
    return [rng.randint(0, 2**32 - 1) for _ in range(num_cycles)]

def hash_string(data: str) -> str:
    """Computes the SHA256 hash of a string."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def run_experiment(
    slice_name: str,
    cycles: int,
    seed: int,
    mode: str,
    out_dir: Path,
    config: Dict[str, Any],
):
    """Main function to run the uplift experiment."""
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
            mock_result = eval(chosen_item) if slice_name == "arithmetic_simple" else f"Expanded({chosen_item})"
            success = success_metric(chosen_item, mock_result)

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

Calibration Mode (--calibration):
- Forces cycles=10 (small, cheap sanity runs)
- Runs both baseline and RFL modes
- Writes to results/uplift_u2/calibration/<slice>/...
- Validates schemas, manifests, and determinism
- Does NOT compute uplift statistics
        """
    )
    parser.add_argument("--slice", required=True, type=str, help="The experiment slice to run (e.g., 'arithmetic_simple').")
    parser.add_argument("--cycles", type=int, help="Number of experiment cycles to run (required unless --calibration).")
    parser.add_argument("--seed", required=True, type=int, help="Initial random seed for deterministic execution.")
    parser.add_argument("--mode", choices=["baseline", "rfl"], help="Execution mode: 'baseline' or 'rfl' (required unless --calibration).")
    parser.add_argument("--out", type=str, help="Output directory for results and manifest files (required unless --calibration).")
    parser.add_argument("--config", default="config/curriculum_uplift_phase2.yaml", type=str, help="Path to the curriculum config file.")
    parser.add_argument("--calibration", action="store_true", help="Run in calibration mode: forces cycles=10, runs both baseline and RFL, validates schemas and determinism.")

    args = parser.parse_args()

    config_path = Path(args.config)

    if args.calibration:
        # Calibration mode: run both baseline and RFL with validation
        from experiments.u2_calibration import CALIBRATION_CYCLES, run_calibration

        out_base = Path("results/uplift_u2/calibration")
        cycles = args.cycles if args.cycles is not None else CALIBRATION_CYCLES

        print("=" * 60)
        print("CALIBRATION FIRE HARNESS")
        print("PHASE II — NOT USED IN PHASE I")
        print("=" * 60)
        print(f"Slice: {args.slice}")
        print(f"Cycles: {cycles}")
        print(f"Seed: {args.seed}")
        print(f"Output: {out_base / args.slice}")
        print()

        summary = run_calibration(
            slice_name=args.slice,
            seed=args.seed,
            config_path=config_path,
            out_base=out_base,
            cycles=cycles,
        )

        # Print summary
        print()
        print("=" * 60)
        print("CALIBRATION SUMMARY")
        print("=" * 60)
        print(f"Slice: {summary['slice']}")
        print(f"Cycles: {summary['cycles']}")
        print(f"Baseline success count: {summary['baseline']['success_count']}")
        print(f"RFL success count: {summary['rfl']['success_count']}")
        print(f"Baseline deterministic: {summary['baseline']['determinism'].get('deterministic', 'N/A') if summary['baseline']['determinism'] else 'N/A'}")
        print(f"RFL deterministic: {summary['rfl']['determinism'].get('deterministic', 'N/A') if summary['rfl']['determinism'] else 'N/A'}")
        print(f"Overall status: {summary['overall_status']}")

        if summary["errors"]:
            print(f"\nErrors: {summary['errors']}")
        if summary["baseline"]["schema_errors"]:
            print(f"Baseline schema errors: {summary['baseline']['schema_errors']}")
        if summary["rfl"]["schema_errors"]:
            print(f"RFL schema errors: {summary['rfl']['schema_errors']}")

        sys.exit(0 if summary["overall_status"] == "passed" else 1)
    else:
        # Regular mode: require all arguments
        if args.cycles is None:
            parser.error("--cycles is required unless --calibration is specified")
        if args.mode is None:
            parser.error("--mode is required unless --calibration is specified")
        if args.out is None:
            parser.error("--out is required unless --calibration is specified")

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
