#!/usr/bin/env python3
# PHASE II — NOT USED IN PHASE I
"""
RFL Policy Tools CLI
====================

Developer tooling for inspecting policy state and configuration.

Usage:
    python -m rfl.policy.tools --slice slice_uplift_goal
    python -m rfl.policy.tools --slice slice_uplift_goal --snapshot path/to/snapshot.json
    python -m rfl.policy.tools --list-slices

This is read-only inspection only — does not modify state or training logic.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_phase2_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the Phase II RFL policy configuration.
    
    Args:
        config_path: Optional path to config file. If None, uses default.
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default path
        config_path = Path("config/rfl_policy_phase2.yaml")
    
    if not config_path.exists():
        # Try alternate location
        alt_path = Path("config/curriculum_uplift_phase2.yaml")
        if alt_path.exists():
            config_path = alt_path
        else:
            return {}
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        print(f"WARNING: Failed to parse config file {config_path}: {e}")
        return {}


def get_learning_schedule_for_slice(
    slice_name: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get learning schedule configuration for a slice.
    
    Args:
        slice_name: Name of the slice
        config: Full configuration dictionary
    
    Returns:
        Learning schedule dictionary
    """
    # Check for per-slice config
    slices_config = config.get("slices", {})
    if slice_name in slices_config:
        slice_cfg = slices_config[slice_name]
        if "learning_schedule" in slice_cfg:
            return slice_cfg["learning_schedule"]
    
    # Fall back to global config
    global_config = config.get("global", {})
    return {
        "learning_rate": global_config.get("learning_rate", 0.01),
        "decay_factor": global_config.get("decay_factor", 0.999),
        "min_learning_rate": global_config.get("min_learning_rate", 0.0001),
        "max_weight_norm_l2": global_config.get("max_weight_norm_l2", 10.0),
        "max_abs_weight": global_config.get("max_abs_weight", 5.0),
    }


def inspect_slice(
    slice_name: str,
    config: Dict[str, Any],
    snapshot_path: Optional[Path] = None,
) -> None:
    """
    Print detailed inspection of a slice's policy configuration.
    
    Args:
        slice_name: Name of the slice to inspect
        config: Configuration dictionary
        snapshot_path: Optional path to policy snapshot file
    """
    # Import here to avoid circular imports
    from .features import SLICE_FEATURE_MASKS
    from .update import (
        LearningScheduleConfig,
        init_cold_start,
        init_from_file,
        summarize_policy_state,
    )
    
    print(f"=" * 60)
    print(f"RFL Policy Inspection: {slice_name}")
    print(f"PHASE II — NOT USED IN PHASE I")
    print(f"=" * 60)
    print()
    
    # 1. Feature mask
    print("Feature Mask (SLICE_FEATURE_MASKS):")
    print("-" * 40)
    feature_mask = SLICE_FEATURE_MASKS.get(slice_name, SLICE_FEATURE_MASKS.get("default", []))
    for i, fname in enumerate(feature_mask, 1):
        print(f"  {i}. {fname}")
    print()
    
    # 2. Learning schedule
    print("Learning Schedule Configuration:")
    print("-" * 40)
    schedule_dict = get_learning_schedule_for_slice(slice_name, config)
    schedule = LearningScheduleConfig.from_dict(schedule_dict)
    for key, value in schedule.to_dict().items():
        print(f"  {key}: {value}")
    print()
    
    # 3. Policy state (from snapshot or cold start)
    print("Policy State:")
    print("-" * 40)
    if snapshot_path and snapshot_path.exists():
        print(f"  Source: {snapshot_path}")
        state = init_from_file(snapshot_path)
    else:
        print(f"  Source: Cold start (no snapshot)")
        state = init_cold_start(slice_name, schedule)
    
    print(f"  Slice name: {state.slice_name}")
    print(f"  Update count: {state.update_count}")
    print(f"  Learning rate: {state.learning_rate}")
    print(f"  Seed: {state.seed}")
    print(f"  Clamped: {state.clamped}")
    print(f"  Clamp count: {state.clamp_count}")
    print(f"  Phase: {state.phase}")
    print()
    
    # 4. Current weights
    print("Current Weights:")
    print("-" * 40)
    if state.weights:
        for fname in sorted(state.weights.keys()):
            print(f"  {fname}: {state.weights[fname]:.6f}")
    else:
        print("  (no weights - zero initialization)")
    print()
    
    # 5. Telemetry summary
    print("Telemetry Summary:")
    print("-" * 40)
    summary = summarize_policy_state(state)
    for key, value in sorted(summary.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print()


def list_slices(config: Dict[str, Any]) -> None:
    """
    List all available slices.
    
    Args:
        config: Configuration dictionary
    """
    from .features import SLICE_FEATURE_MASKS
    
    print("Available Slices:")
    print("-" * 40)
    
    # From config
    slices_config = config.get("slices", {})
    config_slices = set(slices_config.keys())
    
    # From feature masks
    mask_slices = set(SLICE_FEATURE_MASKS.keys())
    
    all_slices = config_slices | mask_slices
    
    for slice_name in sorted(all_slices):
        markers = []
        if slice_name in config_slices:
            markers.append("config")
        if slice_name in mask_slices:
            markers.append("features")
        print(f"  {slice_name} [{', '.join(markers)}]")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RFL Policy Inspection Tool (PHASE II)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
    python -m rfl.policy.tools --slice slice_uplift_goal
    python -m rfl.policy.tools --slice arithmetic_simple --snapshot state.json
    python -m rfl.policy.tools --list-slices
    python -m rfl.policy.tools --slice default --config custom_config.yaml

PHASE II — NOT USED IN PHASE I
        """
    )
    
    parser.add_argument(
        "--slice",
        type=str,
        help="Slice name to inspect",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        help="Path to policy snapshot file (optional)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file (default: config/rfl_policy_phase2.yaml)",
    )
    parser.add_argument(
        "--list-slices",
        action="store_true",
        help="List all available slices",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config) if args.config else None
    config = load_phase2_config(config_path)
    
    if args.list_slices:
        if args.json:
            from .features import SLICE_FEATURE_MASKS
            slices_config = config.get("slices", {})
            output = {
                "slices": list(set(slices_config.keys()) | set(SLICE_FEATURE_MASKS.keys()))
            }
            print(json.dumps(output, indent=2))
        else:
            list_slices(config)
        return 0
    
    if not args.slice:
        parser.print_help()
        print("\nError: --slice is required unless using --list-slices")
        return 1
    
    snapshot_path = Path(args.snapshot) if args.snapshot else None
    
    if args.json:
        # JSON output mode
        from .features import SLICE_FEATURE_MASKS
        from .update import (
            LearningScheduleConfig,
            init_cold_start,
            init_from_file,
            summarize_policy_state,
        )
        
        schedule_dict = get_learning_schedule_for_slice(args.slice, config)
        schedule = LearningScheduleConfig.from_dict(schedule_dict)
        
        if snapshot_path and snapshot_path.exists():
            state = init_from_file(snapshot_path)
        else:
            state = init_cold_start(args.slice, schedule)
        
        output = {
            "slice_name": args.slice,
            "feature_mask": SLICE_FEATURE_MASKS.get(args.slice, SLICE_FEATURE_MASKS.get("default", [])),
            "learning_schedule": schedule.to_dict(),
            "state": state.to_dict(),
            "summary": summarize_policy_state(state),
            "phase": "II",
        }
        print(json.dumps(output, indent=2))
    else:
        inspect_slice(args.slice, config, snapshot_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
