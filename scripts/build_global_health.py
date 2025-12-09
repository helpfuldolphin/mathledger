#!/usr/bin/env python3
"""
Build Global Health Surface — Phase VII NEURAL LINK CI Script

Operation CORTEX: Phase VII Global Health Assembly
==================================================

Assembles global_health.json from component artifacts.

Usage:
    python scripts/build_global_health.py \
        --fm-health artifacts/fm_health.json \
        --tda-snapshot artifacts/tda/governance_snapshot.json \
        --output global_health.json

Exit Codes:
    0: Status is OK
    1: Status is WARN or BLOCK (or error)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure parent packages are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.health.global_builder import build_global_health_surface
from backend.health.canonicalize import canonicalize_global_health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_json_file(path: Path) -> dict:
    """Load JSON file with error handling."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    """
    Main entry point for global health builder.

    Returns:
        Exit code: 0 if status is OK, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Build global health surface from component artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/build_global_health.py --fm-health artifacts/fm_health.json --output global_health.json
    python scripts/build_global_health.py --fm-health fm.json --tda-snapshot tda.json -o global_health.json
        """,
    )

    parser.add_argument(
        "--fm-health",
        type=Path,
        required=True,
        help="Path to FM health JSON file (required)",
    )
    parser.add_argument(
        "--tda-snapshot",
        type=Path,
        default=None,
        help="Path to TDA governance snapshot JSON file (optional)",
    )
    parser.add_argument(
        "--replay-result",
        type=Path,
        default=None,
        help="Path to replay safety result JSON file (optional)",
    )
    parser.add_argument(
        "--learning-report",
        type=Path,
        default=None,
        help="Path to learning health report JSON file (optional)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output path for global_health.json",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress informational output",
    )

    args = parser.parse_args()

    try:
        # Validate FM health file exists
        if not args.fm_health.exists():
            logger.error(f"FM health file not found: {args.fm_health}")
            return 1

        # Load FM health
        fm_health = load_json_file(args.fm_health)
        if not args.quiet:
            logger.info(f"Loaded FM health from {args.fm_health}")

        # Load optional tiles
        tda_snapshot = None
        if args.tda_snapshot and args.tda_snapshot.exists():
            tda_snapshot = load_json_file(args.tda_snapshot)
            if not args.quiet:
                logger.info(f"Loaded TDA snapshot from {args.tda_snapshot}")

        replay_result = None
        if args.replay_result and args.replay_result.exists():
            replay_result = load_json_file(args.replay_result)
            if not args.quiet:
                logger.info(f"Loaded replay result from {args.replay_result}")

        learning_report = None
        if args.learning_report and args.learning_report.exists():
            learning_report = load_json_file(args.learning_report)
            if not args.quiet:
                logger.info(f"Loaded learning report from {args.learning_report}")

        # Build surface
        surface = build_global_health_surface(
            fm_health=fm_health,
            tda_snapshot=tda_snapshot,
            replay_result=replay_result,
            learning_report=learning_report,
        )

        # Convert to dict and canonicalize
        surface_dict = surface.to_dict()
        canonical_dict = canonicalize_global_health(surface_dict)

        # Ensure output directory exists
        args.output.parent.mkdir(parents=True, exist_ok=True)

        # Write output
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(canonical_dict, f, indent=2, sort_keys=True)

        if not args.quiet:
            logger.info(f"Global health surface written to: {args.output}")
            logger.info(f"Status: {surface.status}")

        # Print status to stdout for CI parsing
        print(f"status={surface.status}")

        # Return exit code based on status
        if surface.status == "OK":
            return 0
        else:
            return 1

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
