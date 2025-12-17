#!/usr/bin/env python3
"""
Lean Shadow Inspection Harness

STATUS: PHASE X — SHADOW MODE INSPECTION TOOL

This script accepts a shadow radar JSON input and produces:
- Human-readable summary
- Health tile output
- Artifact file in artifacts/ directory

SHADOW MODE CONTRACT:
- Never blocks execution
- Exit code always 0 (even on errors)
- Purely observational tool
- Compatible with both P3 (synthetic) and P4 (real-runner) modes

Usage:
    python scripts/lean_shadow_inspect.py <radar.json> [--output <tile.json>]

Example:
    python scripts/lean_shadow_inspect.py shadow_radar.json
    python scripts/lean_shadow_inspect.py shadow_radar.json --output artifacts/lean_shadow_tile.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.health.lean_shadow_adapter import (
    build_lean_shadow_tile_for_global_health,
)


def load_radar_from_file(radar_path: Path) -> Dict[str, Any]:
    """Load shadow radar from JSON file."""
    try:
        with open(radar_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Radar file not found: {radar_path}", file=sys.stderr)
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in radar file: {e}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Error: Failed to load radar file: {e}", file=sys.stderr)
        return {}


def print_summary(tile: Dict[str, Any]) -> None:
    """Print human-readable summary of shadow tile."""
    status = tile.get("status", "UNKNOWN")
    error_rate = tile.get("structural_error_rate", 0.0)
    resource_band = tile.get("shadow_resource_band", "UNKNOWN")
    anomalies = tile.get("dominant_anomalies", [])
    headline = tile.get("headline", "No headline available")
    
    print("=" * 70)
    print("LEAN SHADOW MODE INSPECTION SUMMARY")
    print("=" * 70)
    print()
    print(f"Status:           {status}")
    print(f"Error Rate:       {error_rate:.1%}")
    print(f"Resource Band:    {resource_band}")
    print(f"Anomalies:        {len(anomalies)} detected")
    if anomalies:
        print(f"  Top anomalies:  {', '.join(anomalies[:3])}")
    print()
    print(f"Headline:         {headline}")
    print()
    print("=" * 70)


def write_tile_to_file(tile: Dict[str, Any], output_path: Path) -> bool:
    """Write tile to JSON file."""
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tile, f, indent=2, ensure_ascii=False)
        
        print(f"Tile written to: {output_path}")
        return True
    except Exception as e:
        print(f"Warning: Failed to write tile to {output_path}: {e}", file=sys.stderr)
        return False


def main() -> int:
    """Main entry point. Always returns 0 (never blocks)."""
    parser = argparse.ArgumentParser(
        description="Inspect Lean shadow radar and generate health tile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "radar",
        type=Path,
        help="Path to shadow radar JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for tile JSON (default: artifacts/lean_shadow_tile.json)",
    )
    
    args = parser.parse_args()
    
    # Load radar
    radar = load_radar_from_file(args.radar)
    if not radar:
        print("Warning: Empty or invalid radar file. Continuing with empty radar.", file=sys.stderr)
        radar = {
            "structural_error_rate": 0.0,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": [],
            "total_shadow_requests": 0,
        }
    
    # Build tile
    try:
        tile = build_lean_shadow_tile_for_global_health(radar)
    except Exception as e:
        print(f"Warning: Failed to build tile: {e}", file=sys.stderr)
        # Return minimal tile on error
        tile = {
            "schema_version": "1.0.0",
            "status": "OK",
            "structural_error_rate": 0.0,
            "shadow_resource_band": "LOW",
            "dominant_anomalies": [],
            "headline": "Lean shadow inspection encountered an error.",
        }
    
    # Print summary
    print_summary(tile)
    
    # Write tile to file
    if args.output:
        output_path = args.output
    else:
        # Default to artifacts directory
        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        output_path = artifacts_dir / "lean_shadow_tile.json"
    
    write_tile_to_file(tile, output_path)
    
    # Always return 0 (never block)
    return 0


if __name__ == "__main__":
    sys.exit(main())

