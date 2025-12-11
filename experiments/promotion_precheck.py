#!/usr/bin/env python3
"""
PHASE II — Promotion Precheck with TDA Alignment

Advisory-only precheck that loads fused evidence summaries and checks for
TDA alignment issues. This tool detects misalignments between uplift decisions
and TDA outcomes, but does NOT claim "uplift achieved."

Exit Codes:
  0: OK or WARN (no blocking issues)
  1: BLOCK (TDA conflict detected - advisory only)
  2: ERROR (system/configuration error)

Alignment Rules:
  - BLOCK: Any run has PASS uplift but BLOCK TDA (conflicted)
  - WARN: Any run has PASS uplift but low HSS (hidden instability)
  - OK: No conflicts or hidden instability

Note: This is an advisory tool for detecting misalignment, not a certification
      of success. It does not claim uplift has been achieved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evidence_fusion import (
    AlignmentStatus,
    FusedEvidence,
)


def load_fused_evidence(path: Path) -> FusedEvidence:
    """
    Load fused evidence from a JSON file.

    Args:
        path: Path to fused evidence JSON file

    Returns:
        FusedEvidence object

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Fused evidence file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return FusedEvidence.from_dict(data)


def print_alignment_report(fused: FusedEvidence) -> None:
    """
    Print alignment analysis report to stdout.

    Args:
        fused: FusedEvidence to analyze
    """
    print("=" * 80)
    print("PROMOTION PRECHECK — TDA ALIGNMENT ANALYSIS")
    print("=" * 80)
    print()

    print(f"Total runs: {len(fused.runs)}")
    print(f"Alignment status: {fused.tda_alignment.alignment_status.value}")
    print()

    if fused.tda_alignment.alignment_status == AlignmentStatus.OK:
        print("✓ No TDA alignment issues detected")
        print()
        print("All runs are aligned:")
        for run in fused.runs:
            print(f"  • {run.run_id}: "
                  f"uplift={run.uplift.promotion_decision.value}, "
                  f"TDA={run.tda.tda_outcome.value}, "
                  f"HSS={run.tda.HSS:.3f}")

    elif fused.tda_alignment.alignment_status == AlignmentStatus.WARN:
        print("⚠️  WARNING: Hidden instability detected")
        print()
        print("The following runs have PASS uplift but low HSS:")
        for run_id in fused.tda_alignment.hidden_instability_runs:
            run = next(r for r in fused.runs if r.run_id == run_id)
            print(f"  • {run_id}: "
                  f"uplift=PASS, "
                  f"HSS={run.tda.HSS:.3f} (below threshold)")
        print()
        print("This indicates potential hidden state instability that may affect")
        print("reproducibility or reliability. Review TDA metrics before promotion.")

    elif fused.tda_alignment.alignment_status == AlignmentStatus.BLOCK:
        print("✗ BLOCK: Uplift/TDA conflict detected")
        print()
        print("The following runs have PASS uplift but BLOCK TDA:")
        for run_id in fused.tda_alignment.conflicted_runs:
            run = next(r for r in fused.runs if r.run_id == run_id)
            print(f"  • {run_id}: "
                  f"uplift=PASS, "
                  f"TDA=BLOCK, "
                  f"HSS={run.tda.HSS:.3f}, "
                  f"block_rate={run.tda.block_rate:.3f}")
        print()
        print("This is a critical misalignment: uplift metrics suggest promotion,")
        print("but TDA analysis indicates blocking issues. Do NOT promote until")
        print("this conflict is resolved.")

    print()
    print("=" * 80)


def run_precheck(fused_path: Path) -> int:
    """
    Run promotion precheck with TDA alignment analysis.

    Args:
        fused_path: Path to fused evidence JSON file

    Returns:
        Exit code (0 for OK/WARN, 1 for BLOCK, 2 for ERROR)
    """
    try:
        # Load fused evidence
        fused = load_fused_evidence(fused_path)

        # Print alignment report
        print_alignment_report(fused)

        # Determine exit code based on alignment status
        if fused.tda_alignment.alignment_status == AlignmentStatus.BLOCK:
            # Advisory BLOCK: Log as advisory, exit non-zero
            print("Exit code: 1 (advisory BLOCK: TDA conflict)", file=sys.stderr)
            print()
            print("NOTE: This is an advisory block. It does not claim uplift was",
                  file=sys.stderr)
            print("      achieved or denied. It only detects misalignment between",
                  file=sys.stderr)
            print("      uplift metrics and TDA analysis.", file=sys.stderr)
            return 1

        elif fused.tda_alignment.alignment_status == AlignmentStatus.WARN:
            # Warning: Exit zero, but print warning to stderr
            print("Exit code: 0 (OK with warnings)", file=sys.stderr)
            print()
            print("WARNING: Hidden instability detected. Review TDA metrics.",
                  file=sys.stderr)
            return 0

        else:
            # OK: Exit zero
            print("Exit code: 0 (OK)")
            return 0

    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        print(f"ERROR: Invalid fused evidence file: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: Unexpected error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 2


def main() -> int:
    """CLI entry point for promotion precheck."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Promotion precheck with TDA alignment analysis (advisory only)"
    )
    parser.add_argument(
        "fused_evidence",
        type=Path,
        help="Path to fused evidence JSON file",
    )

    args = parser.parse_args()

    return run_precheck(args.fused_evidence)


if __name__ == "__main__":
    sys.exit(main())
