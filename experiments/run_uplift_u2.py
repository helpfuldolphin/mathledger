# PHASE II -- NOT USED IN PHASE I
#
# This script runs a U2 uplift experiment. It is designed to be deterministic
# and self-contained for reproducibility. It supports two modes: 'baseline'
# for random ordering and 'rfl' for policy-driven ordering.
#
# Now uses the modular u2_pipeline for:
# - curriculum_loader_v2: Loading Phase II curriculum configuration
# - feature_extraction + scoring: Feature vectors and policy scoring
# - slice_success_metrics: Integration with success metrics module
# - manifest_generator: Unified manifest generation
# - attestation_bindings: Cycle attestation for traceability
#
# Absolute Safeguards:
# - Do NOT reinterpret Phase I logs as uplift evidence.
# - All Phase II artifacts must be clearly labeled "PHASE II -- NOT USED IN PHASE I".
# - All code must remain deterministic except random shuffle in the baseline policy.
# - RFL uses verifiable feedback only (no RLHF, no preferences, no proxy rewards).
# - All new files must be standalone and MUST NOT modify Phase I behavior.
# - Zero interpretation of uplift -- only raw data and placeholders.

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add project root to sys.path for direct script execution
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml

# Import modular pipeline components
from experiments.u2_pipeline import (
    # Constants
    PHASE_II_LABEL,
    # Curriculum Loader V2
    SliceConfig,
    CurriculumConfigV2,
    load_curriculum_v2,
    # Feature Extraction + Scoring
    CandidateFeatures,
    PolicyWeights,
    compute_item_hash,
    extract_features,
    score_candidate,
    extract_and_score_candidates,
    # Success Metrics
    SuccessMetricConfig,
    SuccessMetricResult,
    evaluate_success_metric,
    # Attestation Bindings
    CycleAttestation,
    compute_attestation_hash,
    create_cycle_attestation,
    # Manifest Generator
    DebugArtifact,
    PairedRunManifest,
    compute_slice_config_hash,
    compute_ht_series_hash,
    generate_seed_schedule,
    create_paired_manifest,
    save_manifest,
    save_debug_artifacts,
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


# --- RFL Policy with Feature-Based Scoring ---
# Enhanced implementation that uses the modular feature extraction and scoring.

class RFLPolicy:
    """
    RFL policy model with feature-based scoring.
    
    Uses PolicyWeights from the u2_pipeline module for candidate scoring.
    Tracks success history for learning.
    """
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.weights = PolicyWeights(
            length_weight=0.0,
            complexity_weight=0.0,
            success_history_weight=0.0,
        )
        self.success_history: Dict[str, float] = {}  # item_hash -> success score
        self.attempt_counts: Dict[str, int] = {}      # item_hash -> attempt count
        self.success_counts: Dict[str, int] = {}      # item_hash -> success count

    def get_weights(self) -> PolicyWeights:
        """Return current policy weights."""
        return self.weights

    def score_with_features(
        self, items: List[str]
    ) -> List[Tuple[str, CandidateFeatures, float]]:
        """
        Score items using feature extraction and policy weights.
        
        Returns list of (item, features, score) tuples sorted by score descending.
        """
        return extract_and_score_candidates(items, self.weights, self.success_history)

    def score(self, items: List[str]) -> List[float]:
        """
        Legacy scoring method for backward compatibility.
        
        Returns list of scores in the same order as input items.
        """
        scored = self.score_with_features(items)
        # Create a lookup for scores
        score_lookup = {item: score for item, _, score in scored}
        return [score_lookup.get(item, 0.0) for item in items]

    def update(self, item: str, success: bool):
        """
        Update the policy based on feedback.
        
        Uses verifiable feedback only -- no RLHF, no preferences.
        """
        item_hash = compute_item_hash(item)
        
        # Update attempt and success counts
        self.attempt_counts[item_hash] = self.attempt_counts.get(item_hash, 0) + 1
        if success:
            self.success_counts[item_hash] = self.success_counts.get(item_hash, 0) + 1
        
        # Compute success rate for this item
        attempts = self.attempt_counts[item_hash]
        successes = self.success_counts.get(item_hash, 0)
        self.success_history[item_hash] = successes / attempts if attempts > 0 else 0.0
        
        # Simple gradient-like weight update
        eta = 0.01  # Learning rate
        if success:
            # Reinforce current strategy
            self.weights = PolicyWeights(
                length_weight=self.weights.length_weight - eta * 0.1,  # Prefer shorter
                complexity_weight=self.weights.complexity_weight + eta * 0.05,
                success_history_weight=self.weights.success_history_weight + eta,
            )
        else:
            # Exploration: small adjustment
            self.weights = PolicyWeights(
                length_weight=self.weights.length_weight + eta * 0.05,
                complexity_weight=self.weights.complexity_weight - eta * 0.025,
                success_history_weight=max(0.0, self.weights.success_history_weight - eta * 0.1),
            )


# --- Core Runner Logic ---

def hash_string(data: str) -> str:
    """Computes the SHA256 hash of a string."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def run_experiment(
    slice_name: str,
    cycles: int,
    seed: int,
    mode: str,
    out_dir: Path,
    curriculum_config: CurriculumConfigV2,
    emit_debug_artifacts: bool = True,
) -> Tuple[List[Dict[str, Any]], List[DebugArtifact]]:
    """
    Main function to run the uplift experiment.
    
    Returns:
        Tuple of (ht_series, debug_artifacts) for manifest generation.
    """
    print(f"--- Running Experiment: slice={slice_name}, mode={mode}, cycles={cycles}, seed={seed} ---")
    print(f"{PHASE_II_LABEL}")

    # 1. Setup using curriculum_loader_v2
    out_dir.mkdir(parents=True, exist_ok=True)
    slice_config = curriculum_config.get_slice(slice_name)
    items = slice_config.items
    
    # Get legacy success metric for backward compatibility
    success_metric = METRIC_DISPATCHER.get(slice_name)
    if not success_metric:
        print(f"ERROR: Success metric for slice '{slice_name}' not found.", file=sys.stderr)
        sys.exit(1)

    seed_schedule = generate_seed_schedule(seed, cycles)
    policy = RFLPolicy(seed) if mode == "rfl" else None
    ht_series: List[Dict[str, Any]] = []
    debug_artifacts: List[DebugArtifact] = []

    results_path = out_dir / f"uplift_u2_{slice_name}_{mode}.jsonl"
    manifest_path = out_dir / f"uplift_u2_manifest_{slice_name}_{mode}.json"

    # 2. Main Loop
    with open(results_path, "w", encoding="utf-8") as results_f:
        for i in range(cycles):
            cycle_seed = seed_schedule[i]
            rng = random.Random(cycle_seed)
            
            # --- Feature Extraction & Ordering Step ---
            candidate_ordering_trace: List[Dict[str, Any]] = []
            feature_vectors: List[Dict[str, Any]] = []
            current_weights: Dict[str, float] = {}
            
            if mode == "baseline":
                # Baseline: random shuffle ordering
                ordered_items = list(items)
                rng.shuffle(ordered_items)
                chosen_item = ordered_items[0]
                
                # Still extract features for debug artifacts
                for idx, item in enumerate(ordered_items):
                    features = extract_features(item)
                    feature_vectors.append(features.to_dict())
                    candidate_ordering_trace.append({
                        "rank": idx,
                        "item": item,
                        "item_hash": features.item_hash,
                        "score": None,  # No score in baseline mode
                        "selection_method": "random_shuffle",
                    })
                current_weights = {"baseline_mode": True}
                
            elif mode == "rfl":
                # RFL: use policy scoring with feature extraction
                scored_items = policy.score_with_features(items)
                chosen_item = scored_items[0][0]
                current_weights = policy.get_weights().to_dict()
                
                # Record feature vectors and ordering trace
                for idx, (item, features, score) in enumerate(scored_items):
                    feature_vectors.append(features.to_dict())
                    candidate_ordering_trace.append({
                        "rank": idx,
                        "item": item,
                        "item_hash": features.item_hash,
                        "score": score,
                        "selection_method": "policy_score",
                    })
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # --- Mock Execution & Evaluation ---
            # In a real system, this would be a call to the substrate.
            mock_result = eval(chosen_item) if slice_name == "arithmetic_simple" else f"Expanded({chosen_item})"
            success = success_metric(chosen_item, mock_result)

            # --- Success Metric Evaluation (using pipeline) ---
            metric_config = SuccessMetricConfig(
                metric_type="sparse",  # Default to sparse for mock evaluation
                parameters={"min_verified": 1},
            )
            metric_result = evaluate_success_metric(
                config=metric_config,
                verified_count=1 if success else 0,
                attempted_count=1,
            )

            # --- RFL Policy Update ---
            if mode == "rfl":
                policy.update(chosen_item, success)

            # --- Create Cycle Attestation ---
            attestation = create_cycle_attestation(
                cycle_index=i,
                slice_name=slice_name,
                mode=mode,
                seed=cycle_seed,
                item=chosen_item,
                result=str(mock_result),
                success=success,
            )

            # --- Telemetry Logging (exact schema preserved) ---
            telemetry_record = {
                "cycle": i,
                "slice": slice_name,
                "mode": mode,
                "seed": cycle_seed,
                "item": chosen_item,
                "result": str(mock_result),
                "success": success,
                "label": PHASE_II_LABEL,
                # Additional fields for traceability
                "item_hash": attestation.item_hash,
                "attestation_hash": attestation.attestation_hash,
            }
            ht_series.append(telemetry_record)
            results_f.write(json.dumps(telemetry_record) + "\n")
            
            # --- Collect Debug Artifacts ---
            if emit_debug_artifacts:
                debug_artifact = DebugArtifact(
                    cycle_index=i,
                    candidate_ordering_trace=candidate_ordering_trace,
                    feature_vectors=feature_vectors,
                    policy_weights=current_weights,
                    success_metric_evaluation=metric_result.to_dict(),
                )
                debug_artifacts.append(debug_artifact)
            
            print(f"Cycle {i+1}/{cycles}: Chose '{chosen_item}', Success: {success}")

    # 3. Single-Mode Manifest Generation (backward compatible)
    slice_config_dict = slice_config.to_dict()
    slice_config_str = json.dumps(slice_config_dict, sort_keys=True)
    slice_config_hash = hash_string(slice_config_str)
    ht_series_str = json.dumps(ht_series, sort_keys=True)
    ht_series_hash = hash_string(ht_series_str)

    manifest = {
        "label": PHASE_II_LABEL,
        "slice": slice_name,
        "mode": mode,
        "cycles": cycles,
        "initial_seed": seed,
        "slice_config_hash": slice_config_hash,
        "prereg_hash": slice_config.prereg_hash or "N/A",
        "ht_series_hash": ht_series_hash,
        "deterministic_seed_schedule": seed_schedule,
        "outputs": {
            "results": str(results_path),
            "manifest": str(manifest_path),
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as manifest_f:
        json.dump(manifest, manifest_f, indent=2)

    # 4. Save Debug Artifacts
    if emit_debug_artifacts and debug_artifacts:
        debug_path = out_dir / f"debug_artifacts_{slice_name}_{mode}.jsonl"
        save_debug_artifacts(debug_artifacts, debug_path)
        print(f"Debug artifacts written to {debug_path}")

    print(f"\n--- Experiment Complete ---")
    print(f"Results written to {results_path}")
    print(f"Manifest written to {manifest_path}")

    return ht_series, debug_artifacts


def run_paired_experiment(
    slice_name: str,
    cycles: int,
    seed: int,
    out_dir: Path,
    curriculum_config: CurriculumConfigV2,
    emit_debug_artifacts: bool = True,
) -> PairedRunManifest:
    """
    Run a paired baseline + RFL experiment and generate unified manifest.
    
    This runs both modes with the same seed for deterministic comparison.
    """
    print(f"=== Running Paired Experiment: slice={slice_name}, cycles={cycles}, seed={seed} ===")
    print(f"{PHASE_II_LABEL}")
    
    slice_config = curriculum_config.get_slice(slice_name)
    
    # Run baseline
    print("\n--- Running Baseline Mode ---")
    baseline_ht_series, baseline_debug = run_experiment(
        slice_name=slice_name,
        cycles=cycles,
        seed=seed,
        mode="baseline",
        out_dir=out_dir,
        curriculum_config=curriculum_config,
        emit_debug_artifacts=emit_debug_artifacts,
    )
    
    # Run RFL
    print("\n--- Running RFL Mode ---")
    rfl_ht_series, rfl_debug = run_experiment(
        slice_name=slice_name,
        cycles=cycles,
        seed=seed,
        mode="rfl",
        out_dir=out_dir,
        curriculum_config=curriculum_config,
        emit_debug_artifacts=emit_debug_artifacts,
    )
    
    # Generate unified paired manifest
    experiment_id = f"uplift_u2_{slice_name}_{seed}"
    baseline_log_path = str(out_dir / f"uplift_u2_{slice_name}_baseline.jsonl")
    rfl_log_path = str(out_dir / f"uplift_u2_{slice_name}_rfl.jsonl")
    
    debug_baseline_path = None
    debug_rfl_path = None
    if emit_debug_artifacts:
        debug_baseline_path = str(out_dir / f"debug_artifacts_{slice_name}_baseline.jsonl")
        debug_rfl_path = str(out_dir / f"debug_artifacts_{slice_name}_rfl.jsonl")
    
    paired_manifest = create_paired_manifest(
        experiment_id=experiment_id,
        slice_config=slice_config,
        cycles=cycles,
        initial_seed=seed,
        baseline_log_path=baseline_log_path,
        rfl_log_path=rfl_log_path,
        baseline_ht_series=baseline_ht_series,
        rfl_ht_series=rfl_ht_series,
        debug_artifacts_baseline_path=debug_baseline_path,
        debug_artifacts_rfl_path=debug_rfl_path,
    )
    
    # Save paired manifest
    paired_manifest_path = out_dir / f"uplift_u2_paired_manifest_{slice_name}.json"
    save_manifest(paired_manifest, paired_manifest_path)
    
    print(f"\n=== Paired Experiment Complete ===")
    print(f"Unified manifest written to {paired_manifest_path}")
    
    return paired_manifest


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
- Zero interpretation of uplift.
        """
    )
    parser.add_argument("--slice", required=True, type=str, 
                        help="The experiment slice to run (e.g., 'arithmetic_simple').")
    parser.add_argument("--cycles", required=True, type=int, 
                        help="Number of experiment cycles to run.")
    parser.add_argument("--seed", required=True, type=int, 
                        help="Initial random seed for deterministic execution.")
    parser.add_argument("--mode", choices=["baseline", "rfl", "paired"], default="paired",
                        help="Execution mode: 'baseline', 'rfl', or 'paired' (default: paired).")
    parser.add_argument("--out", required=True, type=str, 
                        help="Output directory for results and manifest files.")
    parser.add_argument("--config", default="config/curriculum_uplift_phase2.yaml", type=str, 
                        help="Path to the curriculum config file.")
    parser.add_argument("--no-debug", action="store_true",
                        help="Disable emission of debug artifacts.")

    args = parser.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out)
    emit_debug = not args.no_debug

    # Load curriculum using curriculum_loader_v2
    curriculum_config = load_curriculum_v2(config_path)
    print(f"INFO: Loaded curriculum config (hash: {curriculum_config.config_hash[:16]}...)")

    if args.mode == "paired":
        # Run paired experiment (baseline + RFL with unified manifest)
        run_paired_experiment(
            slice_name=args.slice,
            cycles=args.cycles,
            seed=args.seed,
            out_dir=out_dir,
            curriculum_config=curriculum_config,
            emit_debug_artifacts=emit_debug,
        )
    else:
        # Run single-mode experiment
        run_experiment(
            slice_name=args.slice,
            cycles=args.cycles,
            seed=args.seed,
            mode=args.mode,
            out_dir=out_dir,
            curriculum_config=curriculum_config,
            emit_debug_artifacts=emit_debug,
        )


if __name__ == "__main__":
    main()
