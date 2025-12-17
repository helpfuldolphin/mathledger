#!/usr/bin/env python3
"""
PRNG Drift Check — CI Advisory Tool

Computes PRNG governance drift from history and writes governance tile.
This is an advisory tool; exit codes can be used for gating later.

Exit Codes:
    0: Status is not BLOCK (OK or WARN)
    1: Status is BLOCK (governance violations detected)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rfl.prng.governance import (
    build_prng_drift_radar,
    build_prng_governance_tile,
    build_prng_governance_history,
    PRNGGovernanceSnapshot,
    PolicyEvaluation,
    GovernanceStatus,
    ManifestStatus,
    NamespaceIssues,
)


def load_prng_history(history_path: Path) -> Optional[Dict[str, Any]]:
    """Load PRNG governance history from JSON file."""
    if not history_path.exists():
        return None

    try:
        with open(history_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def create_test_history() -> Dict[str, Any]:
    """Create a synthetic test history for demonstration."""
    # Create test snapshots
    snapshots = [
        PRNGGovernanceSnapshot(
            governance_status=GovernanceStatus.OK,
            manifest_status=ManifestStatus.EQUIVALENT,
            namespace_issues=NamespaceIssues(),
        )
        for _ in range(5)
    ]
    policy_evals = [PolicyEvaluation(violations=[]) for _ in range(5)]

    return build_prng_governance_history(snapshots, policy_evaluations=policy_evals)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PRNG Drift Check — Compute drift and write governance tile"
    )
    parser.add_argument(
        "--history",
        type=Path,
        help="Path to PRNG governance history JSON file",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to write prng_governance_tile.json (default: artifacts/)",
    )
    parser.add_argument(
        "--test-context",
        action="store_true",
        help="Use synthetic test data if history file not found",
    )

    args = parser.parse_args()

    # Load or create history
    history: Optional[Dict[str, Any]] = None
    if args.history:
        history = load_prng_history(args.history)

    if history is None and args.test_context:
        history = create_test_history()
        print("⚠️  Using synthetic test history (--test-context)", file=sys.stderr)

    if history is None:
        print(
            f"❌ ERROR: No PRNG governance history found",
            file=sys.stderr,
        )
        if args.history:
            print(f"   Expected: {args.history}", file=sys.stderr)
        print(
            "   Hint: Use --test-context to generate synthetic test data",
            file=sys.stderr,
        )
        return 1

    # Compute radar and tile
    try:
        radar = build_prng_drift_radar(history)
        tile = build_prng_governance_tile(history, radar=radar)

        # Print summary
        status = tile.get("status", "UNKNOWN")
        drift_status = tile.get("drift_status", "UNKNOWN")
        blocking_rules = tile.get("blocking_rules", [])
        headline = tile.get("headline", "PRNG governance: status unknown")

        print(f"PRNG governance: {status} ({drift_status})")
        if blocking_rules:
            print(f"  Blocking rules: {', '.join(blocking_rules)}")
        print(f"  {headline}")

        # Write tile to artifacts directory
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        tile_path = args.artifacts_dir / "prng_governance_tile.json"
        with open(tile_path, "w") as f:
            json.dump(tile, f, indent=2, sort_keys=True)
        print(f"  Tile written: {tile_path}")

        # Exit code: 0 if not BLOCK, 1 if BLOCK
        if status == GovernanceStatus.BLOCK.value:
            return 1
        else:
            return 0

    except Exception as e:
        print(f"❌ ERROR: Failed to compute PRNG drift: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

