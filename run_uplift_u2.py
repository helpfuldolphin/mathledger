"""
PHASE II — NOT USED IN PHASE I
This module simulates the U2 Uplift Runner, adapted for the newest metric engine.
"""
import sys
from typing import Any, Dict, List

# This will fail if the metric engine is not the expected version, which is intended.
from experiments.slice_success_metrics import compute_metric

# Announce compliance on import
print("PHASE II — NOT USED IN PHASE I: Loading ADAPTED U2 Uplift Runner.", file=sys.stderr)

def load_curriculum_slices() -> Dict[str, Dict[str, Any]]:
    """Loads mock experiment slice configurations."""
    return {
        "slice_uplift_goal_1": {
            "metric_kind": "goal_hit",
            "min_goal_hits": 2,
            "min_total_verified": 3,
            "target_hashes": {"h3", "h5"},
        },
        "slice_uplift_sparse_1": {
            "metric_kind": "density",
            "min_verified": 3,
            "max_candidates": 100,
        },
        "slice_uplift_tree_1": {
            "metric_kind": "chain_length",
            "chain_target_hash": "h4",
            "min_chain_length": 3,
        },
        "slice_uplift_dependency_1": {
            "metric_kind": "multi_goal",
            "required_goal_hashes": {"h1", "h4"},
            "min_each_goal": 1,
        },
    }

def get_mock_run_cycle_data() -> Dict[str, Any]:
    """Generates mock data representing the output of a single RFL run cycle."""
    # h1 <- h2 <- h4
    derivations = [
        {"hash": "h1", "text": "...", "premises": []},
        {"hash": "h2", "text": "...", "premises": ["h1"]},
        {"hash": "h3", "text": "...", "premises": []},
        {"hash": "h4", "text": "...", "premises": ["h2"]},
        {"hash": "h5", "text": "...", "premises": []},
    ]
    verified_hashes = {"h1", "h2", "h3", "h4"}
    
    # The new metric engine expects a 'result' object containing derivations
    result_data = {"derivations": derivations}
    
    return {
        "verified_hashes": verified_hashes,
        "candidates_tried": 150,
        "result": result_data,
    }

def run_slice_experiment(slice_name: str, slice_config: Dict[str, Any]) -> None:
    """Simulates running a single experiment slice and evaluating its metric."""
    print(f"--- Running Slice: {slice_name} ---")

    run_data = get_mock_run_cycle_data()

    # The new API takes all config and run data as flat kwargs
    metric_engine_args = {
        "kind": slice_config.pop("metric_kind"),
        **run_data,
        **slice_config,
    }
    
    try:
        success, value, details = compute_metric(**metric_engine_args)
        print(f"  -> Metric Engine Result: success={success}, value={value:.2f}")
        # print(f"     Details: {details}") # Optional: for debugging
    except Exception as e:
        print(f"  -> Metric Engine ERROR: {e}")
    print("-" * (25 + len(slice_name)))


def main():
    """Main entry point to run all simulated experiments."""
    print("====== Starting Mock U2 Uplift Run (Adapted) ======")
    all_slices = load_curriculum_slices()
    for name, config in all_slices.items():
        run_slice_experiment(name, config)
    print("\n====== Mock U2 Uplift Run (Adapted) Complete ======")

if __name__ == "__main__":
    main()