"""
RFL Metabolism Gate - CI Verification Script

CI-gated script that verifies MathLedger's reflexive metabolism:
- Runs 40-run RFL experiment suite
- Computes bootstrap CIs (≥92% coverage, >1.0 uplift)
- Emits [PASS] or [FAIL] with exit codes
- Generates evidence curves

Exit Codes:
    0: PASS - Metabolism verified (coverage≥0.92, uplift>1)
    1: FAIL - Metabolism criteria not met
    2: ERROR - System/configuration error
    3: ABSTAIN - Insufficient data for statistical inference
"""

import sys
import os
from pathlib import Path
import json
import traceback

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.rfl.config import RFLConfig
from backend.rfl.runner import RFLRunner
from backend.rfl.visualizer import RFLVisualizer


def main():
    """Main entry point for RFL gate."""
    print("=" * 80)
    print("RFL METABOLISM GATE - Reflexive Formal Learning Verification")
    print("=" * 80)
    print()

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="CI-gated RFL metabolism verification"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file (optional)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (5 runs instead of 40)"
    )
    parser.add_argument(
        "--no-curves",
        action="store_true",
        help="Skip evidence curve generation"
    )
    args = parser.parse_args()

    # Load configuration
    try:
        if args.config:
            print(f"Loading configuration from: {args.config}")
            config = RFLConfig.from_json(args.config)
        elif args.quick:
            print("Running in QUICK mode (5 runs)")
            from backend.rfl.config import RFL_QUICK_CONFIG
            config = RFL_QUICK_CONFIG
        else:
            print("Loading configuration from environment")
            config = RFLConfig.from_env()

        config.validate()
        print(f"Configuration loaded: {config.experiment_id}")
        print()

    except Exception as e:
        print(f"[ERROR] Configuration failed: {e}")
        traceback.print_exc()
        return 2

    # Run experiment suite
    try:
        runner = RFLRunner(config)
        results = runner.run_all()

    except Exception as e:
        print(f"[ERROR] Experiment execution failed: {e}")
        traceback.print_exc()
        return 2

    # Generate evidence curves
    if not args.no_curves:
        try:
            print("\nGenerating evidence curves...")

            results_path = Path(config.artifacts_dir) / config.results_file
            curves_path = Path(config.artifacts_dir) / config.curves_file

            from backend.rfl.visualizer import generate_quick_summary
            generate_quick_summary(str(results_path), str(curves_path))

            print(f"Evidence curves saved to: {curves_path}")

        except Exception as e:
            print(f"[WARNING] Evidence curve generation failed: {e}")
            traceback.print_exc()
            # Non-fatal, continue to results

    # Print final verdict
    print()
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    # Extract key metrics
    coverage_ci = results.get("coverage", {}).get("bootstrap_ci")
    uplift_ci = results.get("uplift", {}).get("bootstrap_ci")
    metabolism = results.get("metabolism_verification", {})

    if coverage_ci:
        print(f"Coverage: {coverage_ci['point_estimate']:.4f} "
              f"CI=[{coverage_ci['ci_lower']:.4f}, {coverage_ci['ci_upper']:.4f}] "
              f"({coverage_ci['method']})")
    else:
        print("Coverage: ABSTAIN")

    if uplift_ci:
        print(f"Uplift:   {uplift_ci['point_estimate']:.4f} "
              f"CI=[{uplift_ci['ci_lower']:.4f}, {uplift_ci['ci_upper']:.4f}] "
              f"({uplift_ci['method']})")
    else:
        print("Uplift:   ABSTAIN")

    print()
    print(metabolism.get("message", "No metabolism verdict"))
    print("=" * 80)
    print()

    # Determine exit code
    if metabolism.get("passed"):
        print("✓ CI GATE: PASS")
        return 0
    elif coverage_ci and coverage_ci.get("method") == "ABSTAIN":
        print("⊘ CI GATE: ABSTAIN (insufficient data)")
        return 3
    elif uplift_ci and uplift_ci.get("method") == "ABSTAIN":
        print("⊘ CI GATE: ABSTAIN (insufficient data)")
        return 3
    else:
        print("✗ CI GATE: FAIL")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] RFL gate interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n[FATAL ERROR] Unexpected exception: {e}")
        traceback.print_exc()
        sys.exit(2)
