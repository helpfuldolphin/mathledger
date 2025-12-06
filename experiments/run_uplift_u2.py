# PHASE II — NOT USED IN PHASE I
#
# This script runs a U2 uplift experiment. It is designed to be deterministic
# and self-contained for reproducibility. It supports two modes: 'baseline'
# for random ordering and 'rfl' for policy-driven ordering.
#
# Absolute Safeguards:
# - Do NOT reinterpret Phase I logs as uplift evidence.
# - All Phase II artifacts must be clearly labeled "PHASE II — NOT USED IN PHASE I".
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

# Import new policy module for RFL mode
try:
    from rfl.policy import (
        PolicyScorer,
        PolicyUpdater,
        PolicyStateSnapshot,
        LearningScheduleConfig,
        SLICE_FEATURE_MASKS,
        get_feature_mask,
    )
    HAS_POLICY_MODULE = True
except ImportError:
    HAS_POLICY_MODULE = False

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

# Default snapshot interval (log policy state every N cycles)
DEFAULT_SNAPSHOT_INTERVAL = 10


# --- RFL Policy Classes ---

class RFLPolicyLegacy:
    """A mock RFL policy model (legacy fallback)."""
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
    RFL Policy using the new rfl.policy module.
    
    Uses slice-aware feature masks, PolicyScorer for scoring,
    and PolicyUpdater for deterministic updates.
    """
    
    def __init__(
        self,
        seed: int,
        slice_name: str = "default",
        learning_rate: float = 0.01,
    ):
        self.seed = seed
        self.slice_name = slice_name
        self.learning_rate = learning_rate
        
        # Success/attempt history for feature extraction
        self.success_history: Dict[str, int] = {}
        self.attempt_history: Dict[str, int] = {}
        self.total_candidates_seen = 0
        
        # Initialize scorer with slice-aware mask
        self.scorer = PolicyScorer(
            seed=seed,
            slice_name=slice_name,
        )
        
        # Initialize updater
        self.updater = PolicyUpdater(
            initial_weights=self.scorer.get_weights_dict(),
            learning_rate=learning_rate,
            slice_name=slice_name,
            seed=seed,
        )
        
        # Track policy snapshots
        self.snapshots: List[PolicyStateSnapshot] = []
    
    def score(self, items: List[str]) -> List[float]:
        """Score items using slice-aware features."""
        # Update candidate tracking
        for item in items:
            item_hash = hashlib.sha256(item.encode()).hexdigest()
            if item_hash not in self.attempt_history:
                self.total_candidates_seen += 1
        
        # Score batch
        scored = self.scorer.score_batch(
            candidates=items,
            success_history=self.success_history,
            attempt_history=self.attempt_history,
        )
        
        return [sc.score for sc in scored]
    
    def update(self, item: str, success: bool, cycle_index: Optional[int] = None):
        """Update policy based on verified feedback."""
        item_hash = hashlib.sha256(item.encode()).hexdigest()
        
        # Update history
        self.attempt_history[item_hash] = self.attempt_history.get(item_hash, 0) + 1
        if success:
            self.success_history[item_hash] = self.success_history.get(item_hash, 0) + 1
        
        # Compute reward (simple: 1 for success, -1 for failure)
        reward = 1.0 if success else -1.0
        
        # Apply update
        result = self.updater.update(
            reward=reward,
            success=success,
            cycle_index=cycle_index,
        )
        
        # Update scorer weights
        self.scorer.set_weights(result.new_weights)
    
    def get_snapshot(self, cycle_index: int, reward: Optional[float] = None) -> PolicyStateSnapshot:
        """Create and record a policy state snapshot."""
        snap = self.updater.get_snapshot(
            cycle_index=cycle_index,
            reward=reward,
        )
        self.snapshots.append(snap)
        return snap
    
    def get_weights(self) -> Dict[str, float]:
        """Get current policy weights."""
        return self.scorer.get_weights_dict()


# Alias for backward compatibility
RFLPolicy = RFLPolicyLegacy


def load_policy_config(config_path: Path) -> Optional[Dict[str, Any]]:
    """Load policy-specific config if it exists."""
    policy_config_path = config_path.parent / "rfl_policy_phase2.yaml"
    if policy_config_path.exists():
        with open(policy_config_path, "r") as f:
            return yaml.safe_load(f)
    return None


def get_slice_learning_rate(
    policy_config: Optional[Dict[str, Any]],
    slice_name: str,
) -> float:
    """Get learning rate for a slice from config."""
    if policy_config and HAS_POLICY_MODULE:
        schedule = LearningScheduleConfig.from_yaml_config(policy_config)
        return schedule.get_learning_rate(slice_name)
    return 0.01  # Default


def get_snapshot_interval(policy_config: Optional[Dict[str, Any]]) -> int:
    """Get snapshot interval from config."""
    if policy_config:
        snapshots_config = policy_config.get("policy_snapshots", {})
        return snapshots_config.get("snapshot_interval", DEFAULT_SNAPSHOT_INTERVAL)
    return DEFAULT_SNAPSHOT_INTERVAL


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


def map_slice_to_feature_mask(slice_name: str) -> str:
    """
    Map experiment slice name to feature mask slice name.
    
    Handles the mapping from config slice names to SLICE_FEATURE_MASKS keys.
    """
    # Direct mapping for known uplift slices
    slice_mapping = {
        "arithmetic_simple": "slice_uplift_goal",
        "algebra_expansion": "slice_uplift_tree",
    }
    return slice_mapping.get(slice_name, "default")


def run_experiment(
    slice_name: str,
    cycles: int,
    seed: int,
    mode: str,
    out_dir: Path,
    config: Dict[str, Any],
    config_path: Path,
):
    """Main function to run the uplift experiment."""
    print(f"--- Running Experiment: slice={slice_name}, mode={mode}, cycles={cycles}, seed={seed} ---")
    print(f"PHASE II \u2014 NOT USED IN PHASE I")

    # 1. Setup
    out_dir.mkdir(exist_ok=True, parents=True)
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
    
    # Load policy config for per-slice learning rates
    policy_config = load_policy_config(config_path)
    snapshot_interval = get_snapshot_interval(policy_config)
    
    # Initialize policy based on mode
    policy = None
    use_v2_policy = False
    if mode == "rfl":
        # Determine feature mask slice name
        feature_mask_slice = map_slice_to_feature_mask(slice_name)
        learning_rate = get_slice_learning_rate(policy_config, feature_mask_slice)
        
        if HAS_POLICY_MODULE:
            print(f"INFO: Using RFLPolicyV2 with slice-aware features (mask: {feature_mask_slice})")
            print(f"INFO: Learning rate for {slice_name}: {learning_rate}")
            policy = RFLPolicyV2(
                seed=seed,
                slice_name=feature_mask_slice,
                learning_rate=learning_rate,
            )
            use_v2_policy = True
        else:
            print("INFO: rfl.policy module not available, using legacy RFLPolicy")
            policy = RFLPolicyLegacy(seed)
    
    ht_series = []  # History of Telemetry (Ht)
    policy_snapshots = []  # Policy state snapshots

    results_path = out_dir / f"uplift_u2_{slice_name}_{mode}.jsonl"
    manifest_path = out_dir / f"uplift_u2_manifest_{slice_name}_{mode}.json"
    snapshots_path = out_dir / f"uplift_u2_policy_snapshots_{slice_name}_{mode}.jsonl"

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
                if use_v2_policy:
                    policy.update(chosen_item, success, cycle_index=i)
                else:
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
                "label": "PHASE II \u2014 NOT USED IN PHASE I",
            }
            ht_series.append(telemetry_record)
            results_f.write(json.dumps(telemetry_record) + "\n")
            print(f"Cycle {i+1}/{cycles}: Chose '{chosen_item}', Success: {success}")
            
            # --- Policy State Snapshot (periodic) ---
            if mode == "rfl" and use_v2_policy and ((i + 1) % snapshot_interval == 0 or i == cycles - 1):
                reward = 1.0 if success else -1.0
                snap = policy.get_snapshot(cycle_index=i, reward=reward)
                policy_snapshots.append(snap.to_dict())
                print(f"  [Snapshot] Policy state captured at cycle {i+1}")

    # 3. Write policy snapshots (if any)
    if policy_snapshots:
        with open(snapshots_path, "w") as snap_f:
            for snap in policy_snapshots:
                snap_f.write(json.dumps(snap, sort_keys=True) + "\n")
        print(f"Policy snapshots written to {snapshots_path}")

    # 4. Manifest Generation
    slice_config_str = json.dumps(slice_config, sort_keys=True)
    slice_config_hash = hash_string(slice_config_str)
    ht_series_str = json.dumps(ht_series, sort_keys=True)
    ht_series_hash = hash_string(ht_series_str)

    manifest = {
        "label": "PHASE II \u2014 NOT USED IN PHASE I",
        "slice": slice_name,
        "mode": mode,
        "cycles": cycles,
        "initial_seed": seed,
        "slice_config_hash": slice_config_hash,
        "prereg_hash": slice_config.get("prereg_hash", "N/A"),
        "ht_series_hash": ht_series_hash,
        "deterministic_seed_schedule": seed_schedule,
        "policy_module_version": "v2" if use_v2_policy else "legacy",
        "feature_mask_slice": map_slice_to_feature_mask(slice_name) if mode == "rfl" else None,
        "snapshot_count": len(policy_snapshots),
        "outputs": {
            "results": str(results_path),
            "manifest": str(manifest_path),
            "snapshots": str(snapshots_path) if policy_snapshots else None,
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
        config_path=config_path,
    )

if __name__ == "__main__":
    main()
