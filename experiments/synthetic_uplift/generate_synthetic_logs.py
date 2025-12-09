#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Synthetic Log Generator for U2 Uplift Stress-Testing
-----------------------------------------------------

This script generates SYNTHETIC JSONL logs that match the U2 cycle log schema.
All outputs are deterministic given the same seed and configuration.

NOT derived from real derivations; NOT part of Evidence Pack.

Usage:
    python generate_synthetic_logs.py --slice synthetic_easy --mode baseline --cycles 100 --out ./out --seed 42
    python generate_synthetic_logs.py --slice synthetic_shifted --mode rfl --cycles 200 --out ./out --seed 123

Arguments:
    --slice     Name of synthetic slice (synthetic_easy, synthetic_shifted, synthetic_regression)
    --mode      Execution mode: 'baseline' or 'rfl'
    --cycles    Number of cycles to generate
    --out       Output directory for JSONL files
    --seed      Random seed for deterministic generation (default: 42)
    --config    Path to synthetic_slices.yaml (default: auto-detect)

Output Files:
    {out}/synthetic_{slice}_{mode}.jsonl - Cycle logs
    {out}/synthetic_{slice}_{mode}_manifest.json - Generation manifest

==============================================================================
"""

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ==============================================================================
# SAFETY LABEL - Required on all outputs
# ==============================================================================
SAFETY_LABEL = "PHASE II — SYNTHETIC TEST DATA ONLY"


def load_synthetic_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the synthetic slices configuration.
    
    If no path provided, auto-detect relative to this script.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "synthetic_slices.yaml"
    
    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_slice_config(config: Dict[str, Any], slice_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific slice.
    
    Raises:
        SystemExit if slice not found
    """
    slices = config.get("slices", {})
    if slice_name not in slices:
        available = list(slices.keys())
        print(f"ERROR: Slice '{slice_name}' not found.", file=sys.stderr)
        print(f"Available synthetic slices: {available}", file=sys.stderr)
        sys.exit(1)
    
    return slices[slice_name]


def compute_sha256(data: str) -> str:
    """Compute SHA256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def generate_seed_schedule(initial_seed: int, num_cycles: int) -> List[int]:
    """
    Generate deterministic list of seeds for each cycle.
    
    This ensures reproducibility across runs with the same initial seed.
    """
    rng = random.Random(initial_seed)
    return [rng.randint(0, 2**32 - 1) for _ in range(num_cycles)]


class SyntheticOutcomeGenerator:
    """
    Deterministic generator for synthetic success/failure outcomes.
    
    Uses the slice's probability distribution to generate outcomes
    based on formula class and mode (baseline vs rfl).
    """
    
    def __init__(self, slice_config: Dict[str, Any], mode: str, seed: int):
        self.slice_config = slice_config
        self.mode = mode
        self.probabilities = slice_config["probabilities"][mode]
        self.items = slice_config["items"]
        self.rng = random.Random(seed)
        
        # Build item -> class mapping
        self.item_class_map = {
            item["id"]: item["class"] for item in self.items
        }
    
    def get_item_ids(self) -> List[str]:
        """Return list of all item IDs in this slice."""
        return [item["id"] for item in self.items]
    
    def generate_outcome(self, item_id: str, cycle_seed: int) -> Dict[str, Any]:
        """
        Generate a deterministic outcome for an item.
        
        The outcome is determined by:
        1. The item's formula class
        2. The mode's probability for that class
        3. A deterministic PRNG seeded with the cycle seed + item hash
        
        Returns:
            Dict with 'success', 'outcome', 'mock_result' fields
        """
        item_class = self.item_class_map.get(item_id, "class_a")
        prob_success = self.probabilities.get(item_class, 0.5)
        
        # Create deterministic seed from cycle_seed and item_id
        combined_seed = cycle_seed ^ hash(item_id)
        local_rng = random.Random(combined_seed)
        
        # Generate outcome
        roll = local_rng.random()
        success = roll < prob_success
        
        return {
            "success": success,
            "outcome": "VERIFIED" if success else "ABSTAIN",
            "mock_result": {
                "synthetic": True,
                "probability": prob_success,
                "roll": roll,
                "class": item_class,
            }
        }


def select_item_for_cycle(
    items: List[str],
    mode: str,
    cycle_seed: int,
    policy_scores: Optional[Dict[str, float]] = None,
) -> str:
    """
    Select an item for this cycle based on mode.
    
    - baseline: random shuffle selection
    - rfl: policy-scored selection (simulated)
    """
    rng = random.Random(cycle_seed)
    
    if mode == "baseline":
        # Random selection
        selected = rng.choice(items)
    else:
        # RFL mode: use policy scores if available, else random
        if policy_scores:
            # Sort by score descending, pick top
            scored = sorted(
                [(item, policy_scores.get(item, 0.5)) for item in items],
                key=lambda x: x[1],
                reverse=True,
            )
            selected = scored[0][0]
        else:
            # No policy yet - random
            selected = rng.choice(items)
    
    return selected


def update_policy_scores(
    scores: Dict[str, float],
    item: str,
    success: bool,
    seed: int,
) -> Dict[str, float]:
    """
    Update mock RFL policy scores based on outcome.
    
    Simple multiplicative update rule.
    """
    new_scores = scores.copy()
    
    if item not in new_scores:
        new_scores[item] = 0.5
    
    if success:
        new_scores[item] = min(new_scores[item] * 1.1, 0.99)
    else:
        new_scores[item] = max(new_scores[item] * 0.9, 0.01)
    
    return new_scores


def generate_synthetic_logs(
    slice_name: str,
    mode: str,
    cycles: int,
    out_dir: Path,
    seed: int,
    config: Dict[str, Any],
) -> Path:
    """
    Generate synthetic JSONL logs for the specified slice and mode.
    
    Returns:
        Path to the generated JSONL file
    """
    print(f"{'='*60}")
    print(f"{SAFETY_LABEL}")
    print(f"{'='*60}")
    print(f"Generating synthetic logs:")
    print(f"  Slice:  {slice_name}")
    print(f"  Mode:   {mode}")
    print(f"  Cycles: {cycles}")
    print(f"  Seed:   {seed}")
    print(f"  Output: {out_dir}")
    print()
    
    # Setup
    out_dir.mkdir(parents=True, exist_ok=True)
    slice_config = get_slice_config(config, slice_name)
    generator = SyntheticOutcomeGenerator(slice_config, mode, seed)
    seed_schedule = generate_seed_schedule(seed, cycles)
    
    items = generator.get_item_ids()
    policy_scores: Dict[str, float] = {}
    telemetry_series: List[Dict[str, Any]] = []
    
    # Output file paths
    results_path = out_dir / f"synthetic_{slice_name}_{mode}.jsonl"
    manifest_path = out_dir / f"synthetic_{slice_name}_{mode}_manifest.json"
    
    # Statistics tracking
    success_count = 0
    
    # Main generation loop
    with open(results_path, "w", encoding="utf-8") as f:
        for cycle in range(cycles):
            cycle_seed = seed_schedule[cycle]
            
            # Select item
            chosen_item = select_item_for_cycle(
                items=items,
                mode=mode,
                cycle_seed=cycle_seed,
                policy_scores=policy_scores if mode == "rfl" else None,
            )
            
            # Generate outcome
            outcome = generator.generate_outcome(chosen_item, cycle_seed)
            success = outcome["success"]
            
            if success:
                success_count += 1
            
            # Update policy for RFL mode
            if mode == "rfl":
                policy_scores = update_policy_scores(
                    policy_scores, chosen_item, success, cycle_seed
                )
            
            # Build telemetry record (matching U2 schema)
            record = {
                "cycle": cycle,
                "slice": slice_name,
                "mode": mode,
                "seed": cycle_seed,
                "item": chosen_item,
                "result": json.dumps(outcome["mock_result"]),
                "success": success,
                "outcome": outcome["outcome"],
                "proof_found": success,  # Alias for analysis compatibility
                "abstention": not success,  # Alias for analysis compatibility
                "label": SAFETY_LABEL,
                "synthetic": True,  # Explicit synthetic marker
            }
            
            telemetry_series.append(record)
            f.write(json.dumps(record) + "\n")
            
            if (cycle + 1) % 50 == 0 or cycle == cycles - 1:
                rate = success_count / (cycle + 1) * 100
                print(f"  Cycle {cycle + 1}/{cycles}: success_rate={rate:.1f}%")
    
    # Generate manifest
    manifest = {
        "label": SAFETY_LABEL,
        "synthetic": True,
        "slice": slice_name,
        "mode": mode,
        "cycles": cycles,
        "initial_seed": seed,
        "slice_config_hash": compute_sha256(json.dumps(slice_config, sort_keys=True)),
        "prereg_hash": slice_config.get("prereg_hash", "N/A"),
        "telemetry_hash": compute_sha256(json.dumps(telemetry_series, sort_keys=True)),
        "expected_uplift": slice_config.get("expected_uplift", None),
        "expected_direction": slice_config.get("expected_direction", None),
        "seed_schedule": seed_schedule,
        "statistics": {
            "total_cycles": cycles,
            "success_count": success_count,
            "success_rate": success_count / cycles if cycles > 0 else 0.0,
            "abstention_rate": 1.0 - (success_count / cycles) if cycles > 0 else 1.0,
        },
        "outputs": {
            "results": str(results_path),
            "manifest": str(manifest_path),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    # Summary
    print()
    print(f"{'='*60}")
    print("Generation Complete")
    print(f"{'='*60}")
    print(f"  Results:      {results_path}")
    print(f"  Manifest:     {manifest_path}")
    print(f"  Success rate: {manifest['statistics']['success_rate']*100:.2f}%")
    print(f"  Expected direction: {slice_config.get('expected_direction', 'N/A')}")
    print()
    print(f"⚠️  {SAFETY_LABEL}")
    print()
    
    return results_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=f"{SAFETY_LABEL}\n\nGenerate synthetic U2-compatible logs for stress-testing.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Generate baseline logs for easy slice
  python generate_synthetic_logs.py --slice synthetic_easy --mode baseline --cycles 100 --out ./out

  # Generate RFL logs for shifted slice (expected positive uplift)
  python generate_synthetic_logs.py --slice synthetic_shifted --mode rfl --cycles 200 --out ./out

  # Generate with specific seed for reproducibility
  python generate_synthetic_logs.py --slice synthetic_regression --mode baseline --cycles 100 --out ./out --seed 12345
""",
    )
    
    parser.add_argument(
        "--slice",
        required=True,
        type=str,
        help="Synthetic slice name (synthetic_easy, synthetic_shifted, synthetic_regression)",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "rfl"],
        help="Execution mode: 'baseline' or 'rfl'",
    )
    parser.add_argument(
        "--cycles",
        required=True,
        type=int,
        help="Number of cycles to generate",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to synthetic_slices.yaml (default: auto-detect)",
    )
    
    args = parser.parse_args()
    
    # Validate slice name starts with "synthetic_"
    if not args.slice.startswith("synthetic_"):
        print(f"ERROR: Slice name must start with 'synthetic_' (got: {args.slice})", file=sys.stderr)
        print("This ensures synthetic data cannot be confused with real slices.", file=sys.stderr)
        sys.exit(1)
    
    # Load config
    config_path = Path(args.config) if args.config else None
    config = load_synthetic_config(config_path)
    
    # Validate config label
    if config.get("label") != SAFETY_LABEL:
        print(f"ERROR: Config file missing required safety label.", file=sys.stderr)
        sys.exit(1)
    
    # Generate logs
    generate_synthetic_logs(
        slice_name=args.slice,
        mode=args.mode,
        cycles=args.cycles,
        out_dir=Path(args.out),
        seed=args.seed,
        config=config,
    )


if __name__ == "__main__":
    main()

