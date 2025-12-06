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
from typing import Any, Callable, Dict, List, Optional

import yaml

# Import Phase II policy modules
from rfl.policy.update import (
    PolicyStateSnapshot,
    PolicyUpdater,
    LearningScheduleConfig,
    summarize_policy_state,
    init_cold_start,
    init_from_file,
)

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


class RFLPolicyV2:
    """
    Phase II RFL policy with full telemetry and safety guards.
    
    PHASE II — NOT USED IN PHASE I
    """
    def __init__(
        self,
        seed: int,
        slice_name: str,
        schedule: Optional[LearningScheduleConfig] = None,
        initial_state: Optional[PolicyStateSnapshot] = None,
    ):
        """
        Initialize the V2 policy.
        
        Args:
            seed: Random seed for determinism
            slice_name: Name of the curriculum slice
            schedule: Learning schedule config (uses defaults if None)
            initial_state: Optional warm-start state
        """
        self.seed = seed
        self.slice_name = slice_name
        self.schedule = schedule or LearningScheduleConfig()
        self.rng = random.Random(seed)
        
        # Initialize state (cold start or warm start)
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = init_cold_start(slice_name, self.schedule, seed)
        
        # Create updater
        self.updater = PolicyUpdater(self.schedule, self.state)
        
        # Internal score cache for deterministic tie-breaking
        self._score_cache: Dict[str, float] = {}
        self._snapshot_history: List[PolicyStateSnapshot] = []

    def score(self, items: List[str]) -> List[float]:
        """
        Score items using current policy weights.
        
        For Phase II, we use a simple linear model:
            score = w_len * len(item) + w_success * success_rate
        
        Returns scores in same order as items.
        """
        scores = []
        for item in items:
            # Compute basic features
            item_len = len(item)
            item_hash = hashlib.sha256(item.encode('utf-8')).hexdigest()
            
            # Get weights (default to 0 if not set)
            w_len = self.state.weights.get("len", 0.0)
            w_success = self.state.weights.get("success", 0.0)
            
            # Compute success rate from cache
            success_rate = self._score_cache.get(item_hash, 0.5)
            
            # Linear score
            score = w_len * item_len + w_success * success_rate
            
            # Add small deterministic noise for tie-breaking
            # Use item hash to ensure determinism
            noise = (int(item_hash[:8], 16) % 1000) / 100000.0
            score += noise
            
            scores.append(score)
        
        return scores

    def update(self, item: str, success: bool):
        """
        Update policy based on verifiable feedback.
        
        Args:
            item: The item that was evaluated
            success: Whether the evaluation succeeded
        """
        item_hash = hashlib.sha256(item.encode('utf-8')).hexdigest()
        
        # Update success rate cache
        old_rate = self._score_cache.get(item_hash, 0.5)
        alpha = 0.1  # Exponential moving average
        new_rate = (1 - alpha) * old_rate + alpha * (1.0 if success else 0.0)
        self._score_cache[item_hash] = new_rate
        
        # Compute gradient based on success
        gradient = 1.0 if success else -1.0
        
        # Apply update through the updater (with safety guards)
        gradients = {
            "success": gradient * 0.1,  # Small update for success weight
            "len": gradient * -0.01,    # Slight preference for shorter on success
        }
        self.state = self.updater.batch_update(gradients)

    def get_state(self) -> PolicyStateSnapshot:
        """Get current policy state snapshot."""
        return self.state

    def get_summary(self) -> Dict[str, Any]:
        """Get telemetry summary."""
        return summarize_policy_state(self.state)

    def take_snapshot(self) -> PolicyStateSnapshot:
        """Take and store a snapshot of current state."""
        snapshot = PolicyStateSnapshot(
            slice_name=self.state.slice_name,
            weights=dict(self.state.weights),
            update_count=self.state.update_count,
            learning_rate=self.state.learning_rate,
            seed=self.state.seed,
            clamped=self.state.clamped,
            clamp_count=self.state.clamp_count,
            phase=self.state.phase,
        )
        self._snapshot_history.append(snapshot)
        return snapshot

    def export_state(self, path: Path):
        """Export current state to file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, indent=2)


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
    policy_in: Optional[Path] = None,
    policy_out: Optional[Path] = None,
):
    """
    Main function to run the uplift experiment.
    
    PHASE II — NOT USED IN PHASE I
    
    Args:
        slice_name: Name of the experiment slice
        cycles: Number of cycles to run
        seed: Initial random seed
        mode: 'baseline' or 'rfl'
        out_dir: Output directory
        config: Configuration dictionary
        policy_in: Optional path to warm-start policy snapshot
        policy_out: Optional path to export final policy snapshot
    """
    print(f"--- Running Experiment: slice={slice_name}, mode={mode}, cycles={cycles}, seed={seed} ---")
    print(f"PHASE II — NOT USED IN PHASE I")

    # 1. Setup
    out_dir.mkdir(parents=True, exist_ok=True)
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
    
    # Initialize policy based on mode
    policy: Optional[RFLPolicyV2] = None
    policy_v1: Optional[RFLPolicy] = None
    
    if mode == "rfl":
        # Load learning schedule from config
        schedule_config = slice_config.get("learning_schedule", {})
        schedule = LearningScheduleConfig.from_dict(schedule_config) if schedule_config else LearningScheduleConfig()
        
        # Initialize policy state (warm-start or cold-start)
        initial_state: Optional[PolicyStateSnapshot] = None
        if policy_in is not None and policy_in.exists():
            print(f"INFO: Loading policy warm-start from {policy_in}")
            initial_state = init_from_file(policy_in)
            # Validate slice_name matches
            if initial_state.slice_name != slice_name:
                print(f"WARNING: Policy snapshot slice_name '{initial_state.slice_name}' "
                      f"differs from experiment slice '{slice_name}'")
        else:
            print(f"INFO: Using cold-start policy initialization")
        
        # Create V2 policy with full telemetry
        policy = RFLPolicyV2(
            seed=seed,
            slice_name=slice_name,
            schedule=schedule,
            initial_state=initial_state,
        )
        # Also keep V1 for backward compatibility
        policy_v1 = RFLPolicy(seed)
    
    ht_series = []  # History of Telemetry (Hₜ)

    results_path = out_dir / f"uplift_u2_{slice_name}_{mode}.jsonl"
    manifest_path = out_dir / f"uplift_u2_manifest_{slice_name}_{mode}.json"
    
    # Policy telemetry file (PHASE II)
    telemetry_path = out_dir / f"policy_telemetry.jsonl"

    # 2. Main Loop
    with open(results_path, "w") as results_f, open(telemetry_path, "w") as telemetry_f:
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
                # RFL: use policy scoring (V2 with telemetry)
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
                
                # Take periodic snapshot and write telemetry
                if i % 10 == 0 or i == cycles - 1:  # Every 10 cycles and at end
                    snapshot = policy.take_snapshot()
                    policy_summary = summarize_policy_state(snapshot)
                    
                    # Write policy telemetry
                    telemetry_record = {
                        "cycle": i,
                        "mode": mode,
                        "slice": slice_name,
                        "policy_summary": policy_summary,
                        "phase": "II",
                        "label": "PHASE II — NOT USED IN PHASE I",
                    }
                    telemetry_f.write(json.dumps(telemetry_record, sort_keys=True) + "\n")

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

    # 3. Export final policy state if requested
    if mode == "rfl" and policy_out is not None:
        print(f"INFO: Exporting final policy state to {policy_out}")
        policy_out.parent.mkdir(parents=True, exist_ok=True)
        policy.export_state(policy_out)

    # 4. Manifest Generation
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
            "policy_telemetry": str(telemetry_path) if mode == "rfl" else None,
            "policy_out": str(policy_out) if policy_out else None,
        }
    }

    with open(manifest_path, "w") as manifest_f:
        json.dump(manifest, manifest_f, indent=2)

    print(f"\n--- Experiment Complete ---")
    print(f"Results written to {results_path}")
    print(f"Manifest written to {manifest_path}")
    if mode == "rfl":
        print(f"Policy telemetry written to {telemetry_path}")

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

PHASE II — NOT USED IN PHASE I
        """
    )
    parser.add_argument("--slice", required=True, type=str, help="The experiment slice to run (e.g., 'arithmetic_simple').")
    parser.add_argument("--cycles", required=True, type=int, help="Number of experiment cycles to run.")
    parser.add_argument("--seed", required=True, type=int, help="Initial random seed for deterministic execution.")
    parser.add_argument("--mode", required=True, choices=["baseline", "rfl"], help="Execution mode: 'baseline' or 'rfl'.")
    parser.add_argument("--out", required=True, type=str, help="Output directory for results and manifest files.")
    parser.add_argument("--config", default="config/curriculum_uplift_phase2.yaml", type=str, help="Path to the curriculum config file.")
    
    # Policy warm-start / cold-start options (PHASE II)
    parser.add_argument(
        "--policy-in",
        type=str,
        default=None,
        help="Optional path to warm-start policy snapshot file. If provided, loads policy state from this file."
    )
    parser.add_argument(
        "--policy-out",
        type=str,
        default=None,
        help="Optional path to export final policy snapshot. If provided, writes final policy state to this file."
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out)
    policy_in = Path(args.policy_in) if args.policy_in else None
    policy_out = Path(args.policy_out) if args.policy_out else None

    config = get_config(config_path)

    run_experiment(
        slice_name=args.slice,
        cycles=args.cycles,
        seed=args.seed,
        mode=args.mode,
        out_dir=out_dir,
        config=config,
        policy_in=policy_in,
        policy_out=policy_out,
    )

if __name__ == "__main__":
    main()
