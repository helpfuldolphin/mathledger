"""
PHASE-II -- NOT USED IN PHASE I

U2 Uplift Experiment CLI
========================

This script runs a U2 uplift experiment. It is designed to be deterministic
and self-contained for reproducibility. It supports two modes:
    - ``baseline``: Random ordering (seeded shuffle)
    - ``rfl``: Policy-driven ordering with verifiable feedback

**Determinism Notes:**
    - All random operations use seeded RNG instances.
    - Same seed always produces the same experiment results.
    - Seed schedules are pre-computed for full reproducibility.

Absolute Safeguards:
    - Do NOT reinterpret Phase I logs as uplift evidence.
    - All Phase II artifacts must be clearly labeled "PHASE II -- NOT USED IN PHASE I".
    - All code must remain deterministic except random shuffle in the baseline policy.
    - RFL uses verifiable feedback only (no RLHF, no preferences, no proxy rewards).
    - All new files must be standalone and MUST NOT modify Phase I behavior.

Usage:
    python run_uplift_u2.py --slice=arithmetic_simple --cycles=100 --seed=42 --mode=baseline --out=/tmp/results
    python run_uplift_u2.py --slice=arithmetic_simple --cycles=100 --seed=42 --mode=rfl --out=/tmp/results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Import from the structured u2 module for reusable components
# These provide the core experiment functionality
from experiments.u2.seed import generate_seed_schedule as _generate_seed_schedule
from experiments.u2.metrics import (
    metric_arithmetic_simple,
    metric_algebra_expansion,
    METRIC_DISPATCHER,
    MetricFunction,
)
from experiments.u2.manifest import compute_hash as hash_string
from experiments.u2.runner import RFLPolicy


def get_config(config_path: Path) -> Dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The parsed configuration dictionary.

    Raises:
        SystemExit: If the config file is not found.

    **Determinism Notes:**
        - Configuration loading is deterministic (same file = same result).
    """
    print(f"INFO: Loading config from {config_path}")
    if not config_path.exists():
        print(
            f"ERROR: Config file not found at {config_path}. "
            f"Ensure the file exists and the path is correct.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_seed_schedule(initial_seed: int, num_cycles: int) -> List[int]:
    """Generate a deterministic list of seeds for each cycle.

    This function wraps the u2 module's implementation for backward compatibility.

    Args:
        initial_seed: The initial seed value.
        num_cycles: The number of cycles (seeds) to generate.

    Returns:
        A list of ``num_cycles`` integer seeds.

    **Determinism Notes:**
        - Same initial_seed always produces the same schedule.
    """
    return _generate_seed_schedule(initial_seed, num_cycles)


def run_experiment(
    slice_name: str,
    cycles: int,
    seed: int,
    mode: str,
    out_dir: Path,
    config: Dict[str, Any],
) -> None:
    """Run the U2 uplift experiment.

    This function orchestrates the complete experiment execution including:
    - Configuration validation
    - Seed schedule generation
    - Cycle execution (baseline or RFL mode)
    - Result and manifest generation

    Args:
        slice_name: The experiment slice to run (e.g., "arithmetic_simple").
        cycles: Number of experiment cycles to run. Must be positive.
        seed: Initial random seed for deterministic execution.
        mode: Execution mode ("baseline" or "rfl").
        out_dir: Output directory for results and manifest files.
        config: Configuration dictionary containing slice definitions.

    Raises:
        SystemExit: On critical errors (slice not found, metric not found).
        ValueError: If mode is invalid.

    **Determinism Notes:**
        - Same inputs always produce the same outputs.
        - Seed schedule is pre-computed for reproducibility.
        - Policy updates (RFL mode) are deterministic.
    """
    print(f"--- Running Experiment: slice={slice_name}, mode={mode}, cycles={cycles}, seed={seed} ---")
    print("PHASE II -- NOT USED IN PHASE I")

    # Validate mode
    if mode not in ("baseline", "rfl"):
        raise ValueError(
            f"Invalid mode '{mode}'. Expected 'baseline' or 'rfl'. "
            f"Baseline uses random ordering; RFL uses policy-driven ordering."
        )

    # Setup output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get slice configuration
    slice_config = config.get("slices", {}).get(slice_name)
    if not slice_config:
        available_slices = list(config.get("slices", {}).keys())
        print(
            f"ERROR: Slice '{slice_name}' not found in config. "
            f"Available slices: {available_slices}",
            file=sys.stderr,
        )
        sys.exit(1)

    items: List[str] = slice_config.get("items", [])
    if not items:
        print(
            f"ERROR: Slice '{slice_name}' has no items defined. "
            f"Add 'items' list to the slice configuration.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Get success metric
    success_metric: Optional[MetricFunction] = METRIC_DISPATCHER.get(slice_name)
    if not success_metric:
        print(
            f"ERROR: Success metric for slice '{slice_name}' not found. "
            f"Register it using experiments.u2.metrics.register_metric().",
            file=sys.stderr,
        )
        sys.exit(1)

    # Generate deterministic seed schedule
    seed_schedule = generate_seed_schedule(seed, cycles)

    # Initialize policy (RFL mode only)
    policy: Optional[RFLPolicy] = RFLPolicy(seed) if mode == "rfl" else None

    # Telemetry series (Ht)
    ht_series: List[Dict[str, Any]] = []

    # Output paths
    results_path = out_dir / f"uplift_u2_{slice_name}_{mode}.jsonl"
    manifest_path = out_dir / f"uplift_u2_manifest_{slice_name}_{mode}.json"

    # Main execution loop
    with open(results_path, "w", encoding="utf-8") as results_f:
        for i in range(cycles):
            cycle_seed = seed_schedule[i]
            rng = random.Random(cycle_seed)

            # Select item based on mode
            if mode == "baseline":
                # Baseline: random shuffle ordering
                ordered_items = list(items)
                rng.shuffle(ordered_items)
                chosen_item = ordered_items[0]
            else:  # mode == "rfl"
                # RFL: use policy scoring
                item_scores = policy.score(items)
                scored_items = sorted(
                    zip(items, item_scores), key=lambda x: x[1], reverse=True
                )
                chosen_item = scored_items[0][0]

            # Mock execution & evaluation
            if slice_name == "arithmetic_simple":
                try:
                    mock_result = eval(chosen_item)
                except Exception:
                    mock_result = None
            else:
                mock_result = f"Expanded({chosen_item})"

            success = success_metric(chosen_item, mock_result)

            # RFL policy update
            if mode == "rfl" and policy is not None:
                policy.update(chosen_item, success)

            # Telemetry logging
            telemetry_record: Dict[str, Any] = {
                "cycle": i,
                "slice": slice_name,
                "mode": mode,
                "seed": cycle_seed,
                "item": chosen_item,
                "result": str(mock_result),
                "success": success,
                "label": "PHASE II -- NOT USED IN PHASE I",
            }
            ht_series.append(telemetry_record)
            results_f.write(json.dumps(telemetry_record, sort_keys=True) + "\n")
            print(f"Cycle {i + 1}/{cycles}: Chose '{chosen_item}', Success: {success}")

    # Manifest generation
    slice_config_str = json.dumps(slice_config, sort_keys=True)
    slice_config_hash = hash_string(slice_config_str)
    ht_series_str = json.dumps(ht_series, sort_keys=True)
    ht_series_hash = hash_string(ht_series_str)

    manifest: Dict[str, Any] = {
        "label": "PHASE II -- NOT USED IN PHASE I",
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
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as manifest_f:
        json.dump(manifest, manifest_f, indent=2, sort_keys=True)

    print(f"\n--- Experiment Complete ---")
    print(f"Results written to {results_path}")
    print(f"Manifest written to {manifest_path}")


def main() -> None:
    """CLI entry point for U2 uplift experiments.

    Parses command-line arguments and runs the experiment with the specified
    configuration. Supports both baseline (random) and RFL (policy-driven) modes.

    **Determinism Notes:**
        - Same CLI arguments always produce the same results.
    """
    parser = argparse.ArgumentParser(
        description="PHASE II U2 Uplift Runner. Must not be used for Phase I.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Absolute Safeguards:
- Do NOT reinterpret Phase I logs as uplift evidence.
- All Phase II artifacts must be clearly labeled.
- RFL uses verifiable feedback only.
        """,
    )
    parser.add_argument(
        "--slice",
        required=True,
        type=str,
        help="The experiment slice to run (e.g., 'arithmetic_simple').",
    )
    parser.add_argument(
        "--cycles",
        required=True,
        type=int,
        help="Number of experiment cycles to run.",
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Initial random seed for deterministic execution.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "rfl"],
        help="Execution mode: 'baseline' or 'rfl'.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="Output directory for results and manifest files.",
    )
    parser.add_argument(
        "--config",
        default="config/curriculum_uplift_phase2.yaml",
        type=str,
        help="Path to the curriculum config file.",
    )

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
