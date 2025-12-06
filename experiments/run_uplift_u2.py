#!/usr/bin/env python3
"""
PHASE II -- NOT RUN IN PHASE I

U2 Uplift Experiment Runner
============================

Runs a paired baseline and RFL experiment on a chosen uplift slice.
Writes logs and a preliminary manifest, but leaves analysis and uplift
claims to other tools.

Usage:
    uv run python experiments/run_uplift_u2.py \\
      --slice-name=slice_uplift_sparse \\
      --cycles=500 \\
      --seed=1234 \\
      --out-dir=results/uplift_u2/slice_uplift_sparse

Outputs:
    - baseline.jsonl    : Cycle logs from baseline mode
    - rfl.jsonl         : Cycle logs from RFL mode
    - experiment_manifest.json : Experiment metadata (NO uplift claims)

Absolute Safeguards:
    - Do NOT reinterpret Phase I logs as uplift evidence.
    - All Phase II artifacts must be clearly labeled "PHASE II -- NOT RUN IN PHASE I".
    - Do NOT compute uplift stats or claim any improvement.
    - Do NOT modify Phase I configs or logs.
    - RFL uses verifiable feedback only (no RLHF, no preferences, no proxy rewards).
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.run_fo_cycles import CycleRunner
from curriculum.gates import load

# Constants
PHASE_LABEL = "PHASE II -- NOT RUN IN PHASE I"


def generate_seed_schedule(
    global_seed: int, slice_name: str, num_cycles: int
) -> List[int]:
    """
    Generate a deterministic seed schedule from (global_seed, slice_name, cycle_index).

    Each per-cycle seed is derived by hashing the combination of global_seed,
    slice_name, and cycle_index to ensure reproducibility.

    Args:
        global_seed: The initial seed for the experiment
        slice_name: Name of the slice being run
        num_cycles: Number of cycles to generate seeds for

    Returns:
        List of deterministic seeds, one per cycle
    """
    seeds = []
    for cycle_index in range(num_cycles):
        # Derive per-cycle seed from (global_seed, slice_name, cycle_index)
        seed_material = f"{global_seed}:{slice_name}:{cycle_index}"
        seed_hash = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
        # Take first 8 hex chars (32 bits) and convert to int
        cycle_seed = int(seed_hash[:8], 16)
        seeds.append(cycle_seed)
    return seeds


def compute_config_hash(slice_name: str, system: str = "pl") -> Optional[str]:
    """
    Compute SHA256 hash of the slice configuration from curriculum.yaml.

    Args:
        slice_name: Name of the slice
        system: System slug (default: pl)

    Returns:
        SHA256 hex digest of the slice config, or None if not found
    """
    from dataclasses import asdict, is_dataclass

    try:
        system_cfg = load(system)
        for slice_obj in system_cfg.slices:
            if slice_obj.name == slice_name:
                # Convert gates to dict if it's a dataclass
                gates_dict = (
                    asdict(slice_obj.gates)
                    if is_dataclass(slice_obj.gates)
                    else slice_obj.gates
                )
                # Serialize slice config deterministically
                config_dict = {
                    "name": slice_obj.name,
                    "params": slice_obj.params,
                    "gates": gates_dict,
                }
                config_str = json.dumps(config_dict, sort_keys=True)
                return hashlib.sha256(config_str.encode("utf-8")).hexdigest()
    except Exception:
        pass
    return None


def get_success_metric_kind(slice_name: str, system: str = "pl") -> Optional[str]:
    """
    Get the success_metric_kind from the slice config in curriculum.yaml.

    Args:
        slice_name: Name of the slice
        system: System slug (default: pl)

    Returns:
        Success metric kind string, or None if not configured
    """
    try:
        system_cfg = load(system)
        for slice_obj in system_cfg.slices:
            if slice_obj.name == slice_name:
                # Check for explicit success_metric_kind in params
                if "success_metric_kind" in slice_obj.params:
                    return slice_obj.params["success_metric_kind"]
                # Default metric kinds based on slice characteristics
                # For uplift slices, default to "derivation_verified_count"
                if "uplift" in slice_name:
                    return "derivation_verified_count"
                return "proof_found"
    except Exception:
        pass
    return None


def run_uplift_u2(
    slice_name: str,
    cycles: int,
    seed: int,
    out_dir: Path,
    system: str = "pl",
) -> Dict[str, Any]:
    """
    Run a paired baseline and RFL experiment on the specified slice.

    Args:
        slice_name: Name of the curriculum slice to use
        cycles: Number of cycles to run for each mode
        seed: Global seed for deterministic execution
        out_dir: Output directory for results and manifest
        system: System slug (default: pl)

    Returns:
        Manifest dictionary with experiment metadata
    """
    print("=" * 60)
    print(f"{PHASE_LABEL}")
    print("U2 UPLIFT EXPERIMENT")
    print("=" * 60)
    print(f"Slice: {slice_name}")
    print(f"Cycles per mode: {cycles}")
    print(f"Seed: {seed}")
    print(f"Output directory: {out_dir}")
    print()

    # Setup output paths
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = out_dir / "baseline.jsonl"
    rfl_path = out_dir / "rfl.jsonl"
    manifest_path = out_dir / "experiment_manifest.json"

    # Generate deterministic seed schedule shared by both modes
    seed_schedule = generate_seed_schedule(seed, slice_name, cycles)

    # Record start time
    started_at = datetime.now(timezone.utc).isoformat()

    # --- Run Baseline Mode ---
    print("Running baseline mode...")
    try:
        baseline_runner = CycleRunner(
            mode="baseline",
            output_path=baseline_path,
            slice_name=slice_name,
            system=system,
        )
        baseline_runner.run(cycles)
        print(f"Baseline complete -> {baseline_path}")
    except Exception as e:
        print(f"ERROR during baseline run: {e}", file=sys.stderr)
        raise

    # --- Run RFL Mode ---
    print("\nRunning RFL mode...")
    try:
        rfl_runner = CycleRunner(
            mode="rfl",
            output_path=rfl_path,
            slice_name=slice_name,
            system=system,
        )
        rfl_runner.run(cycles)
        print(f"RFL complete -> {rfl_path}")
    except Exception as e:
        print(f"ERROR during RFL run: {e}", file=sys.stderr)
        raise

    # Record completion time
    completed_at = datetime.now(timezone.utc).isoformat()

    # Compute slice config hash
    slice_config_hash = compute_config_hash(slice_name, system)

    # Get success metric kind
    success_metric_kind = get_success_metric_kind(slice_name, system)

    # --- Create Manifest ---
    # Note: NO uplift decision - leave as null/pending for analysis tools
    manifest: Dict[str, Any] = {
        "label": PHASE_LABEL,
        "manifest_version": "1.0",
        "experiment_id": "uplift_u2",
        "slice_name": slice_name,
        "cycles": cycles,
        "seed": seed,
        "system": system,
        "paths": {
            "baseline_log": str(baseline_path),
            "rfl_log": str(rfl_path),
        },
        "seed_schedule": seed_schedule,
        "slice_config_hash": slice_config_hash,
        "success_metric_kind": success_metric_kind,
        "execution": {
            "started_at": started_at,
            "completed_at": completed_at,
            "executor": "run_uplift_u2.py",
        },
        # Uplift decision fields - explicitly null/pending
        # Analysis tools will fill these in
        "uplift_decision": None,
        "uplift_stats": None,
        "outcome": "pending",
    }

    # Write manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Baseline:  {baseline_path}")
    print(f"RFL:       {rfl_path}")
    print(f"Manifest:  {manifest_path}")
    print()
    print("Note: Uplift decision is PENDING.")
    print("      Run analysis tools to compute statistics and make claims.")

    return manifest


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=f"{PHASE_LABEL}\nU2 Uplift Experiment Runner - runs paired baseline/RFL experiments.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Absolute Safeguards:
- Do NOT reinterpret Phase I logs as uplift evidence.
- All Phase II artifacts are clearly labeled.
- This script does NOT compute uplift stats or claim improvement.
- RFL uses verifiable feedback only.

Example:
    uv run python experiments/run_uplift_u2.py \\
      --slice-name=slice_uplift_sparse \\
      --cycles=500 \\
      --seed=1234 \\
      --out-dir=results/uplift_u2/slice_uplift_sparse
        """,
    )
    parser.add_argument(
        "--slice-name",
        required=True,
        type=str,
        help="Curriculum slice name from config/curriculum.yaml",
    )
    parser.add_argument(
        "--cycles",
        required=True,
        type=int,
        help="Number of cycles to run for each mode (baseline and RFL)",
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Global seed for deterministic execution",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=str,
        help="Output directory for results and manifest files",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="pl",
        help="System slug (default: pl)",
    )

    args = parser.parse_args()

    run_uplift_u2(
        slice_name=args.slice_name,
        cycles=args.cycles,
        seed=args.seed,
        out_dir=Path(args.out_dir),
        system=args.system,
    )


if __name__ == "__main__":
    main()
