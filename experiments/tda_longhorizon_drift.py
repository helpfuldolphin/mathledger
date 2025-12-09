#!/usr/bin/env python3
"""
TDA Long-Horizon Drift Watcher — Phase V Analysis Tool

Operation CORTEX: Phase V Operator Console & Self-Audit Harness
================================================================

Analyzes TDA governance behavior over multiple runs to detect
long-term drift in hard gate calibration and behavior.

Usage:
    python experiments/tda_longhorizon_drift.py \
        --tiles artifacts/tda/governance_tiles/*.json \
        --output artifacts/tda/longhorizon_report.json

    python experiments/tda_longhorizon_drift.py \
        --manifest artifacts/tda/tile_manifest.json \
        --output artifacts/tda/longhorizon_report.json

Inputs:
    - Directory or glob of TDA governance tiles from multiple runs
    - Or a manifest file listing tile paths

Output:
    JSON report with:
    - block_rate_trend: "increasing" | "stable" | "decreasing"
    - mean_hss_trend: "improving" | "stable" | "degrading"
    - golden_alignment_trend: "stable" | "drifting" | "broken"
    - exception_usage: statistics on exception window usage
    - governance_signal: "OK" | "ATTENTION" | "ALERT"
    - recommendations: neutral, structural hints
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Ensure parent packages are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.tda.governance_console import (
    build_longhorizon_drift_report,
    LongHorizonDriftReport,
    LONGHORIZON_DRIFT_SCHEMA_VERSION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_tile(path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a single governance tile from JSON file.

    Args:
        path: Path to tile JSON file.

    Returns:
        Parsed tile dictionary, or None if loading fails.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Normalize tile format
        tile = normalize_tile(data, path)
        return tile

    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in {path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error loading {path}: {e}")
        return None


def normalize_tile(data: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """
    Normalize tile data to expected format.

    Handles various tile formats:
    - Direct evidence tile
    - Wrapped in "tda_governance" key
    - Run summary with TDA fields

    Args:
        data: Raw tile data.
        path: Source path for metadata extraction.

    Returns:
        Normalized tile dictionary.
    """
    # If wrapped in tda_governance
    if "tda_governance" in data:
        tile = data["tda_governance"]
    elif "tda_tile" in data:
        tile = data["tda_tile"]
    elif "evidence" in data and "tda_tile" in data["evidence"]:
        tile = data["evidence"]["tda_tile"]
    else:
        tile = data

    # Ensure required fields with defaults
    normalized = {
        "block_rate": tile.get("block_rate", 0.0),
        "mean_hss": tile.get("mean_hss", 0.0),
        "golden_alignment": tile.get("golden_alignment", "ALIGNED"),
        "exception_active": tile.get("exception_active", False),
        "cycle_count": tile.get("cycle_count", 0),
        "block_count": tile.get("block_count", 0),
        "warn_count": tile.get("warn_count", 0),
        "ok_count": tile.get("ok_count", 0),
        "mode": tile.get("mode", tile.get("hard_gate_mode", "hard")),
    }

    # Extract timestamp if available
    timestamp = tile.get("timestamp") or tile.get("generated_at") or data.get("timestamp")
    if timestamp:
        normalized["timestamp"] = timestamp
    else:
        # Try to extract from filename (e.g., run_20251209_123456.json)
        try:
            stem = path.stem
            if "_" in stem and len(stem) > 15:
                # Assume format: prefix_YYYYMMDD_HHMMSS
                parts = stem.split("_")
                for i, part in enumerate(parts):
                    if len(part) == 8 and part.isdigit():
                        date_part = part
                        time_part = parts[i + 1] if i + 1 < len(parts) and len(parts[i + 1]) == 6 else "000000"
                        normalized["timestamp"] = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}T{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}Z"
                        break
        except Exception:
            pass

    return normalized


def load_tiles_from_glob(pattern: str) -> List[Dict[str, Any]]:
    """
    Load tiles from glob pattern.

    Args:
        pattern: Glob pattern for tile files.

    Returns:
        List of loaded tiles.
    """
    paths = sorted(glob.glob(pattern))
    logger.info(f"Found {len(paths)} files matching pattern: {pattern}")

    tiles = []
    for path in paths:
        tile = load_tile(Path(path))
        if tile:
            tiles.append(tile)

    logger.info(f"Successfully loaded {len(tiles)} tiles")
    return tiles


def load_tiles_from_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """
    Load tiles from manifest file.

    Manifest format:
    {
        "tiles": ["path/to/tile1.json", "path/to/tile2.json", ...]
    }
    or
    ["path/to/tile1.json", "path/to/tile2.json", ...]

    Args:
        manifest_path: Path to manifest JSON file.

    Returns:
        List of loaded tiles.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if isinstance(manifest, list):
        paths = manifest
    elif "tiles" in manifest:
        paths = manifest["tiles"]
    else:
        raise ValueError("Manifest must be a list or have 'tiles' key")

    # Resolve paths relative to manifest
    base_dir = manifest_path.parent
    tiles = []
    for path_str in paths:
        path = Path(path_str)
        if not path.is_absolute():
            path = base_dir / path

        tile = load_tile(path)
        if tile:
            tiles.append(tile)

    logger.info(f"Loaded {len(tiles)} tiles from manifest")
    return tiles


def load_tiles_from_directory(directory: Path) -> List[Dict[str, Any]]:
    """
    Load all JSON tiles from a directory.

    Args:
        directory: Path to directory containing tile files.

    Returns:
        List of loaded tiles.
    """
    pattern = str(directory / "*.json")
    return load_tiles_from_glob(pattern)


def generate_report(
    tiles: Sequence[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> LongHorizonDriftReport:
    """
    Generate long-horizon drift report from tiles.

    Args:
        tiles: Sequence of governance tiles.
        output_path: Optional path to write report.

    Returns:
        LongHorizonDriftReport with analysis.
    """
    report = build_longhorizon_drift_report(tiles)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report.to_json(indent=2))
        logger.info(f"Report written to: {output_path}")

    return report


def print_summary(report: LongHorizonDriftReport) -> None:
    """
    Print human-readable summary of report.

    Args:
        report: LongHorizonDriftReport to summarize.
    """
    print("\n" + "=" * 60)
    print("TDA LONG-HORIZON DRIFT ANALYSIS")
    print("=" * 60)

    print(f"\nRuns Analyzed: {report.runs_analyzed}")
    if report.first_run_timestamp:
        print(f"Period: {report.first_run_timestamp} to {report.last_run_timestamp}")

    print(f"\nGovernance Signal: {report.governance_signal}")

    print("\nTrends:")
    print(f"  Block Rate:        {report.block_rate_trend}")
    print(f"  Mean HSS:          {report.mean_hss_trend}")
    print(f"  Golden Alignment:  {report.golden_alignment_trend}")

    print("\nException Window Usage:")
    print(f"  Total Windows:     {report.exception_usage['total_windows']}")
    print(f"  Per-Run Mean:      {report.exception_usage['per_run_mean']:.4f}")
    print(f"  Trend:             {report.exception_usage['trend']}")

    if report.recommendations:
        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")

    if report.metrics:
        print("\nMetrics:")
        print(f"  Block Rate Mean:   {report.metrics.get('block_rate_mean', 0.0):.4f}")
        print(f"  Block Rate Std:    {report.metrics.get('block_rate_std', 0.0):.4f}")
        print(f"  Mean HSS Mean:     {report.metrics.get('mean_hss_mean', 0.0):.4f}")
        print(f"  Mean HSS Std:      {report.metrics.get('mean_hss_std', 0.0):.4f}")

    print("\n" + "=" * 60)


def main() -> int:
    """
    Main entry point for long-horizon drift analysis.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Analyze TDA governance drift over multiple runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze tiles from glob pattern
    python experiments/tda_longhorizon_drift.py \\
        --tiles "artifacts/tda/*.json" \\
        --output report.json

    # Analyze tiles from manifest
    python experiments/tda_longhorizon_drift.py \\
        --manifest artifacts/tda/manifest.json \\
        --output report.json

    # Analyze all tiles in a directory
    python experiments/tda_longhorizon_drift.py \\
        --directory artifacts/tda/tiles/ \\
        --output report.json
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--tiles",
        type=str,
        help="Glob pattern for tile files (e.g., 'artifacts/tda/*.json')",
    )
    input_group.add_argument(
        "--manifest",
        type=Path,
        help="Path to manifest file listing tile paths",
    )
    input_group.add_argument(
        "--directory",
        type=Path,
        help="Directory containing tile JSON files",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path for JSON report",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Print human-readable summary (default: true)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress summary output (JSON only)",
    )
    parser.add_argument(
        "--json-stdout",
        action="store_true",
        help="Print JSON report to stdout",
    )

    args = parser.parse_args()

    try:
        # Load tiles
        if args.tiles:
            tiles = load_tiles_from_glob(args.tiles)
        elif args.manifest:
            if not args.manifest.exists():
                logger.error(f"Manifest not found: {args.manifest}")
                return 1
            tiles = load_tiles_from_manifest(args.manifest)
        elif args.directory:
            if not args.directory.is_dir():
                logger.error(f"Directory not found: {args.directory}")
                return 1
            tiles = load_tiles_from_directory(args.directory)
        else:
            logger.error("No input source specified")
            return 1

        if not tiles:
            logger.error("No tiles loaded")
            return 1

        # Generate report
        report = generate_report(tiles, args.output)

        # Output
        if not args.quiet:
            print_summary(report)

        if args.json_stdout:
            print(report.to_json())

        return 0

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
