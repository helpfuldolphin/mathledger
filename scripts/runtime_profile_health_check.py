#!/usr/bin/env python3
"""
Runtime Profile Health Check — CI Advisory Script

SHADOW MODE: This script runs the chaos harness and produces a health tile
for the global health dashboard. It is purely advisory and never blocks CI.

This script:
- Runs chaos harness with fixed profile and seed
- Produces a runtime_profile_health.json tile
- Prints warnings to stderr if health is poor (advisory only)
- Always exits with code 0 (non-blocking)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

try:
    from experiments.u2_runtime_chaos import run_chaos_harness
    from experiments.u2.runtime import (
        summarize_runtime_profile_health_for_global_console,
    )
except ImportError as e:
    print(f"ERROR: Failed to import runtime profile modules: {e}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Runtime Profile Health Check — CI Advisory Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="prod-hardened",
        help="Runtime profile to check (default: prod-hardened)",
    )
    parser.add_argument(
        "--env-context",
        type=str,
        choices=["dev", "ci", "prod"],
        default="ci",
        help="Environment context (default: ci)",
    )
    parser.add_argument(
        "--flip-flags",
        type=int,
        default=2,
        help="Number of flags to flip per run (default: 2)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of chaos runs (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed for determinism (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/runtime_profile_health",
        help="Output directory for health tile (default: artifacts/runtime_profile_health)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Optional baseline chaos summary JSON to compare against",
    )

    args = parser.parse_args()

    # Run chaos harness
    try:
        chaos_summary = run_chaos_harness(
            profile_name=args.profile,
            env_context=args.env_context,
            flip_flags=args.flip_flags,
            runs=args.runs,
            seed=args.seed,
        )
    except Exception as e:
        print(f"ERROR: Chaos harness failed: {e}", file=sys.stderr)
        # Create error tile
        chaos_summary = {
            "error": str(e),
            "schema_version": "1.0.0",
            "profile_name": args.profile,
        }

    # Convert to health tile
    try:
        health_tile = summarize_runtime_profile_health_for_global_console(chaos_summary)
    except Exception as e:
        print(f"ERROR: Failed to build health tile: {e}", file=sys.stderr)
        # Create error tile
        health_tile = {
            "schema_version": "1.0.0",
            "tile_type": "runtime_profile_health",
            "status_light": "RED",
            "profile_name": args.profile,
            "profile_stability": 0.0,
            "no_run_rate": 1.0,
            "headline": f"Health tile build failed: {e}",
            "notes": [],
        }

    # Check for baseline comparison
    if args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            try:
                with open(baseline_path, "r", encoding="utf-8") as f:
                    baseline_summary = json.load(f)

                # Compare profile stability
                baseline_stability = baseline_summary.get("profile_stability", 0.0)
                current_stability = chaos_summary.get("profile_stability", 0.0)

                # Compare NO_RUN rate
                baseline_actions = baseline_summary.get("actions", {})
                baseline_no_run = baseline_actions.get("NO_RUN", 0)
                baseline_total = baseline_summary.get("total_runs", 1)
                baseline_no_run_rate = baseline_no_run / baseline_total if baseline_total > 0 else 0.0

                current_actions = chaos_summary.get("actions", {})
                current_no_run = current_actions.get("NO_RUN", 0)
                current_total = chaos_summary.get("total_runs", 1)
                current_no_run_rate = current_no_run / current_total if current_total > 0 else 0.0

                # Detect significant drift
                stability_drop = baseline_stability - current_stability
                no_run_increase = current_no_run_rate - baseline_no_run_rate

                if stability_drop > 0.1 or no_run_increase > 0.1:
                    print(
                        f"WARNING: Profile drift detected (advisory only)",
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

    # Print warnings if health is poor (advisory only)
    status_light = health_tile.get("status_light", "UNKNOWN")
    no_run_rate = health_tile.get("no_run_rate", 0.0)

    if status_light == "RED":
        print(
            f"WARNING: Runtime profile health is RED (advisory only)",
            file=sys.stderr,
        )
        print(
            f"  Profile: {health_tile.get('profile_name', 'unknown')}",
            file=sys.stderr,
        )
        print(
            f"  Stability: {health_tile.get('profile_stability', 0.0):.1%}",
            file=sys.stderr,
        )
        print(f"  NO_RUN rate: {no_run_rate:.1%}", file=sys.stderr)

    if no_run_rate >= 0.2:
        print(
            f"WARNING: High NO_RUN rate: {no_run_rate:.1%} (advisory only)",
            file=sys.stderr,
        )

    # Write health tile
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "runtime_profile_health.json"

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(health_tile, f, indent=2)

        print(f"Runtime profile health tile written to: {output_path}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed to write health tile: {e}", file=sys.stderr)
        # Continue anyway - advisory mode

    # Always exit with 0 (shadow/advisory mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())

