#!/usr/bin/env python3
"""
U2 Runtime Chaos Harness — Shadow-Mode Stress Testing

Advisory chaos harness for stress-testing runtime profiles under random
flag combinations. All outputs are diagnostic and advisory only.

This tool:
- Generates synthetic flag configurations
- Evaluates profiles against these configurations
- Derives fail-safe actions
- Emits deterministic JSON summaries for analysis

SHADOW MODE: This tool never blocks experiments or modifies runtime state.
It is purely observational and designed to feed into dashboards and CI logs.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from experiments.u2.runtime import (
    RuntimeProfile,
    SYNTHETIC_FEATURE_FLAGS,
    load_runtime_profile,
    build_runtime_health_snapshot,
    validate_flag_policy,
    evaluate_runtime_profile,
    derive_runtime_fail_safe_action,
)


def generate_random_flag_config(
    prng: random.Random,
    flip_flags: int,
    safe_flags: List[str],
) -> Dict[str, bool]:
    """
    Generate a random flag configuration.

    Args:
        prng: Seeded random number generator
        flip_flags: Number of flags to randomly flip
        safe_flags: List of safe flag names to choose from

    Returns:
        Dictionary mapping flag names to boolean values
    """
    # Start with defaults
    config = {name: flag.default for name, flag in SYNTHETIC_FEATURE_FLAGS.items()}

    # Randomly flip N flags
    if flip_flags > 0 and safe_flags:
        flags_to_flip = prng.sample(safe_flags, min(flip_flags, len(safe_flags)))
        for flag_name in flags_to_flip:
            config[flag_name] = not config[flag_name]

    return config


def run_chaos_harness(
    profile_name: str,
    env_context: str,
    flip_flags: int,
    runs: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Run the chaos harness against a profile.

    Args:
        profile_name: Name of profile to stress-test
        env_context: Environment context (dev/ci/prod)
        flip_flags: Number of flags to flip per run
        runs: Number of runs to execute
        seed: PRNG seed for determinism

    Returns:
        JSON-serializable chaos summary
    """
    # Load profile
    try:
        profile = load_runtime_profile(profile_name)
    except ValueError as e:
        return {
            "error": str(e),
            "schema_version": "1.0.0",
        }

    # Initialize PRNG with seed
    prng = random.Random(seed)

    # Get safe synthetic flags (all flags in registry)
    safe_flags = list(SYNTHETIC_FEATURE_FLAGS.keys())

    # Track actions across runs
    actions = Counter()
    all_violations = Counter()

    # Run M iterations
    for run_idx in range(runs):
        # Generate random flag config
        flag_config = generate_random_flag_config(prng, flip_flags, safe_flags)

        # Build snapshot
        snapshot = build_runtime_health_snapshot(active_flags=flag_config)

        # Validate policy
        policy_result = validate_flag_policy(env_context, active_flags=flag_config)

        # Evaluate profile
        profile_eval = evaluate_runtime_profile(profile, snapshot, policy_result)

        # Derive fail-safe action
        fail_safe = derive_runtime_fail_safe_action(profile_eval)

        # Record action
        action = fail_safe.get("action", "ALLOW")
        actions[action] += 1

        # Record violations (for frequency analysis)
        violations = profile_eval.get("violations", [])
        for violation in violations:
            all_violations[violation] += 1

    # Compute profile stability (fraction of ALLOW actions)
    total_runs = sum(actions.values())
    profile_stability = actions["ALLOW"] / total_runs if total_runs > 0 else 0.0

    # Get top violations
    top_violations = [violation for violation, _ in all_violations.most_common(5)]

    return {
        "schema_version": "1.0.0",
        "profile_name": profile_name,
        "env_context": env_context,
        "total_runs": runs,
        "seed": seed,
        "flip_flags": flip_flags,
        "actions": dict(actions),
        "profile_stability": round(profile_stability, 4),
        "top_violations": top_violations,
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="U2 Runtime Chaos Harness — Shadow-Mode Stress Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="Runtime profile name to stress-test (e.g., dev-default, ci-strict, prod-hardened)",
    )
    parser.add_argument(
        "--env-context",
        type=str,
        choices=["dev", "ci", "prod"],
        default="dev",
        help="Environment context for policy validation",
    )
    parser.add_argument(
        "--flip-flags",
        type=int,
        default=2,
        help="Number of random flags to flip per run",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of chaos runs to execute",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed for deterministic runs",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path (default: stdout)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Optional baseline chaos summary JSON to compare against for drift detection",
    )

    args = parser.parse_args()

    # Run chaos harness
    summary = run_chaos_harness(
        profile_name=args.profile,
        env_context=args.env_context,
        flip_flags=args.flip_flags,
        runs=args.runs,
        seed=args.seed,
    )

    # Compare with baseline if provided
    if args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            try:
                with open(baseline_path, "r", encoding="utf-8") as f:
                    baseline_summary = json.load(f)

                # Compare profile stability
                baseline_stability = baseline_summary.get("profile_stability", 0.0)
                current_stability = summary.get("profile_stability", 0.0)

                # Compare NO_RUN rate
                baseline_actions = baseline_summary.get("actions", {})
                baseline_no_run = baseline_actions.get("NO_RUN", 0)
                baseline_total = baseline_summary.get("total_runs", 1)
                baseline_no_run_rate = baseline_no_run / baseline_total if baseline_total > 0 else 0.0

                current_actions = summary.get("actions", {})
                current_no_run = current_actions.get("NO_RUN", 0)
                current_total = summary.get("total_runs", 1)
                current_no_run_rate = current_no_run / current_total if current_total > 0 else 0.0

                # Detect significant drift
                stability_drop = baseline_stability - current_stability
                no_run_increase = current_no_run_rate - baseline_no_run_rate

                if stability_drop > 0.1 or no_run_increase > 0.1:
                    print(
                        f"NOTE: Profile drift detected (advisory only)",
                        file=sys.stderr,
                    )
                    print(
                        f"  Stability: {baseline_stability:.1%} -> {current_stability:.1%} (drop: {stability_drop:.1%})",
                        file=sys.stderr,
                    )
                    print(
                        f"  NO_RUN rate: {baseline_no_run_rate:.1%} -> {current_no_run_rate:.1%} (increase: {no_run_increase:.1%})",
                        file=sys.stderr,
                    )
            except Exception as e:
                print(
                    f"WARNING: Failed to compare with baseline: {e}",
                    file=sys.stderr,
                )

    # Output results
    output_json = json.dumps(summary, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json, encoding="utf-8")
        print(f"Chaos summary written to: {output_path}", file=sys.stderr)
    else:
        print(output_json)

    # Exit code based on profile stability (advisory only, not blocking)
    if "error" in summary:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

