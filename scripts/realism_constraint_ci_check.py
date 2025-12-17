#!/usr/bin/env python3
"""
==============================================================================
PHASE X — SHADOW MODE ONLY
==============================================================================

Realism Constraint CI Check
----------------------------

This script runs realism constraint analysis and writes results to JSON.
It always exits with code 0 (SHADOW mode - no gating).

Usage:
    python scripts/realism_constraint_ci_check.py [--ratchet FILE] [--console FILE] [--configs FILE] [--output FILE]

If files are not provided, uses mock data for testing.

==============================================================================
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.synthetic_uplift.constraint_solver import (
    analyze_realism_from_ratchet,
    build_complete_constraint_analysis,
)


def load_json_file(filepath: Path) -> Dict[str, Any]:
    """Load JSON file, return empty dict if file doesn't exist."""
    if not filepath.exists():
        print(f"Warning: {filepath} does not exist, using empty dict", file=sys.stderr)
        return {}
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}", file=sys.stderr)
        return {}


def create_mock_data() -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Create mock data for testing when real data is not available."""
    mock_ratchet = {
        "scenario_retention_score": {
            "scenario_test": 0.7,
        },
        "stability_class": {
            "scenario_test": "STABLE",
        },
    }
    
    mock_console = {
        "slices_to_recalibrate": [],
        "calibration_status": "OK",
        "realism_pressure": 0.1,
    }
    
    mock_configs = {
        "scenario_test": {
            "parameters": {
                "probabilities": {
                    "baseline": {"class_a": 0.70},
                    "rfl": {"class_a": 0.75},
                },
                "drift": {"mode": "none"},
                "correlation": {"rho": 0.5},
                "variance": {"per_cycle_sigma": 0.10, "per_item_sigma": 0.05},
            }
        }
    }
    
    return mock_ratchet, mock_console, mock_configs


def main() -> int:
    """Main entry point for CI check."""
    parser = argparse.ArgumentParser(
        description="Run realism constraint analysis (SHADOW mode - always exits 0)"
    )
    parser.add_argument(
        "--ratchet",
        type=Path,
        help="Path to ratchet JSON file (optional, uses mock if not provided)",
    )
    parser.add_argument(
        "--console",
        type=Path,
        help="Path to calibration console JSON file (optional, uses mock if not provided)",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        help="Path to scenario configs JSON file (optional, uses mock if not provided)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/realism_constraints.json"),
        help="Output path for realism constraints JSON (default: artifacts/realism_constraints.json)",
    )
    
    args = parser.parse_args()
    
    # Load data or use mocks
    if args.ratchet:
        ratchet = load_json_file(args.ratchet)
    else:
        ratchet = {}
    
    if args.console:
        console = load_json_file(args.console)
    else:
        console = {}
    
    if args.configs:
        configs = load_json_file(args.configs)
    else:
        configs = {}
    
    # Use mock data if any required data is missing
    if not ratchet or not console or not configs:
        print("Using mock data (some inputs missing)", file=sys.stderr)
        mock_ratchet, mock_console, mock_configs = create_mock_data()
        ratchet = ratchet or mock_ratchet
        console = console or mock_console
        configs = configs or mock_configs
    
    # Run analysis
    try:
        analysis = analyze_realism_from_ratchet(
            ratchet=ratchet,
            console=console,
            scenario_configs=configs,
        )
    except Exception as e:
        print(f"Error running analysis: {e}", file=sys.stderr)
        # In SHADOW mode, we still exit 0 even on errors
        # Write minimal output
        analysis = {
            "label": "PHASE X — SHADOW MODE ONLY",
            "schema_version": "constraint_solver_v1.0",
            "error": str(e),
            "constraint_solution": {},
            "coupling_map": {},
            "director_tile": {},
        }
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Write output
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        print(f"Realism constraints written to {args.output}", file=sys.stderr)
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        # Still exit 0 in SHADOW mode
    
    # Always exit 0 (SHADOW mode - no gating)
    return 0


if __name__ == "__main__":
    sys.exit(main())

