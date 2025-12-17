#!/usr/bin/env python3
"""
P5 Structural Drill Runner Script

Runs the STRUCTURAL_BREAK drill as an optional stress diagnostic during CAL-EXP-3.
Produces drill artifacts into the specified results directory for evidence pack inclusion.

SHADOW MODE: All drill operations are observational only. No enforcement actions.

Usage:
    python scripts/run_p5_structural_drill.py --output-dir results/p5_drill
    python scripts/run_p5_structural_drill.py --output-dir results/p5_drill --scenario DRILL-SB-001
    python scripts/run_p5_structural_drill.py --output-dir results/p5_drill --sample-rate 10

See: docs/system_law/P5_Structural_Drill_Package.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.dag.structural_drill_runner import (
    DrillArtifact,
    generate_drift_timeline_plot_data,
    run_structural_drill,
)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_SCENARIO_ID = "DRILL-SB-001"
DEFAULT_SAMPLE_RATE = 10
STRUCTURAL_DRILL_ARTIFACT = "structural_drill_artifact.json"
STRUCTURAL_DRILL_TIMELINE = "structural_drill_timeline.json"
STRUCTURAL_DRILL_SUMMARY = "structural_drill_summary.json"
STRUCTURAL_DRILL_PLOT = "structural_drill_plot_data.json"


# =============================================================================
# Drill Execution
# =============================================================================

def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_drill(
    output_dir: Path,
    scenario_id: str = DEFAULT_SCENARIO_ID,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Execute P5 structural drill and write artifacts.

    SHADOW MODE: All operations are observational only.

    Args:
        output_dir: Directory to write artifacts
        scenario_id: Drill scenario identifier
        sample_rate: Cycle sampling rate
        verbose: Print progress messages

    Returns:
        Dict with drill results and artifact paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[STRUCTURAL DRILL] Starting scenario {scenario_id}")
        print(f"[STRUCTURAL DRILL] Output: {output_dir}")
        print(f"[STRUCTURAL DRILL] Sample rate: {sample_rate}")

    # Run drill
    artifact = run_structural_drill(
        scenario_id=scenario_id,
        sample_rate=sample_rate,
        output_dir=None,  # We'll write custom artifacts
    )

    if verbose:
        print(f"[STRUCTURAL DRILL] Drill ID: {artifact.drill_id}")
        print(f"[STRUCTURAL DRILL] Phases completed: {len(artifact.phases)}")
        print(f"[STRUCTURAL DRILL] Cycles sampled: {len(artifact.cycle_results)}")

    # Write main artifact
    artifact_path = output_dir / STRUCTURAL_DRILL_ARTIFACT
    artifact_dict = artifact.to_dict()
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact_dict, f, indent=2)

    # Write timeline data
    timeline_path = output_dir / STRUCTURAL_DRILL_TIMELINE
    timeline_data = {
        "drill_id": artifact.drill_id,
        "scenario_id": artifact.scenario_id,
        "cycles": [r.cycle for r in artifact.cycle_results],
        "cohesion_scores": [r.signal.cohesion_score for r in artifact.cycle_results],
        "patterns": [r.pattern for r in artifact.cycle_results],
        "severities": [r.severity for r in artifact.cycle_results],
        "streaks": [r.streak for r in artifact.cycle_results],
        "phases": [r.phase_name for r in artifact.cycle_results],
        "admissible": [r.signal.admissible for r in artifact.cycle_results],
    }
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline_data, f, indent=2)

    # Write summary
    summary_path = output_dir / STRUCTURAL_DRILL_SUMMARY
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(artifact.summary, f, indent=2)

    # Write plot data
    plot_path = output_dir / STRUCTURAL_DRILL_PLOT
    plot_data = generate_drift_timeline_plot_data(artifact)
    with open(plot_path, "w", encoding="utf-8") as f:
        json.dump(plot_data, f, indent=2)

    # Compute hashes
    artifact_hash = compute_file_hash(artifact_path)
    timeline_hash = compute_file_hash(timeline_path)
    summary_hash = compute_file_hash(summary_path)
    plot_hash = compute_file_hash(plot_path)

    if verbose:
        print(f"[STRUCTURAL DRILL] Artifacts written:")
        print(f"  - {artifact_path} (sha256:{artifact_hash[:16]}...)")
        print(f"  - {timeline_path} (sha256:{timeline_hash[:16]}...)")
        print(f"  - {summary_path} (sha256:{summary_hash[:16]}...)")
        print(f"  - {plot_path} (sha256:{plot_hash[:16]}...)")

    # Build result
    result = {
        "success": True,
        "drill_id": artifact.drill_id,
        "scenario_id": artifact.scenario_id,
        "drill_success": artifact.summary.get("drill_success", False),
        "max_streak": artifact.summary.get("max_streak", 0),
        "break_events": artifact.summary.get("break_events", []),
        "pattern_counts": artifact.summary.get("pattern_counts", {}),
        "artifacts": {
            "main": {
                "path": str(artifact_path),
                "sha256": artifact_hash,
            },
            "timeline": {
                "path": str(timeline_path),
                "sha256": timeline_hash,
            },
            "summary": {
                "path": str(summary_path),
                "sha256": summary_hash,
            },
            "plot": {
                "path": str(plot_path),
                "sha256": plot_hash,
            },
        },
        "shadow_mode": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write drill result manifest
    manifest_path = output_dir / "structural_drill_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"[STRUCTURAL DRILL] Drill success: {artifact.summary.get('drill_success')}")
        print(f"[STRUCTURAL DRILL] Max streak: {artifact.summary.get('max_streak')}")
        print(f"[STRUCTURAL DRILL] Complete.")

    return result


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run P5 Structural Drill (SHADOW MODE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_p5_structural_drill.py --output-dir results/p5_drill
    python scripts/run_p5_structural_drill.py -o results/p5_drill -s DRILL-SB-002 -r 5
    python scripts/run_p5_structural_drill.py -o results/p5_drill --verbose

SHADOW MODE: All drill operations are observational only. No enforcement actions.
        """,
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Output directory for drill artifacts",
    )
    parser.add_argument(
        "-s", "--scenario",
        type=str,
        default=DEFAULT_SCENARIO_ID,
        help=f"Drill scenario ID (default: {DEFAULT_SCENARIO_ID})",
    )
    parser.add_argument(
        "-r", "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Cycle sampling rate (default: {DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress messages",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON to stdout",
    )

    args = parser.parse_args()

    try:
        result = run_drill(
            output_dir=Path(args.output_dir),
            scenario_id=args.scenario,
            sample_rate=args.sample_rate,
            verbose=args.verbose,
        )

        if args.json:
            print(json.dumps(result, indent=2))

        return 0 if result.get("success") else 1

    except Exception as e:
        if args.verbose:
            print(f"[STRUCTURAL DRILL] Error: {e}", file=sys.stderr)
        if args.json:
            print(json.dumps({"success": False, "error": str(e)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
