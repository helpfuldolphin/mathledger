#!/usr/bin/env python3
"""
Phase X P5: Mock vs Real Telemetry Comparison Script

Compares P4 shadow run results between mock and real telemetry adapters.
See docs/system_law/First_Light_P5_Adapter_Checklist.md for specification.

SHADOW MODE CONTRACT:
- This script performs pure comparison (read-only analysis)
- No exit-code gating (SHADOW mode = no enforcement)
- Outputs JSON summary for evidence pack

Usage:
    python scripts/compare_p4_mock_vs_real.py \
        --mock-dir results/first_light_p4/p4_mock_run \
        --real-dir results/first_light_p4/p4_real_run

Outputs:
    JSON comparison summary to stdout (can be piped to file)

Status: P5 POC IMPLEMENTATION
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare P4 mock vs real telemetry runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mock-dir",
        type=str,
        required=True,
        help="Path to mock telemetry P4 run output directory",
    )
    parser.add_argument(
        "--real-dir",
        type=str,
        required=True,
        help="Path to real telemetry P4 run output directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output to stderr",
    )
    return parser.parse_args()


def load_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load p4_summary.json from a run directory.

    Args:
        run_dir: Path to run directory

    Returns:
        Parsed summary dict, or None if not found
    """
    summary_path = run_dir / "p4_summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        return json.load(f)


def load_config(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load run_config.json from a run directory.

    Args:
        run_dir: Path to run directory

    Returns:
        Parsed config dict, or None if not found
    """
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        return json.load(f)


def compare_runs(
    mock_dir: Path,
    real_dir: Path,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compare mock and real telemetry P4 runs.

    SHADOW MODE: Pure comparison, no gating.

    Args:
        mock_dir: Path to mock run directory
        real_dir: Path to real run directory
        verbose: Print verbose output

    Returns:
        Comparison result dict
    """
    result: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "comparison_type": "mock_vs_real",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS",
        "errors": [],
    }

    # Load mock summary
    mock_summary = load_summary(mock_dir)
    if mock_summary is None:
        result["status"] = "ERROR"
        result["errors"].append(f"Mock summary not found: {mock_dir / 'p4_summary.json'}")
        return result

    # Load real summary
    real_summary = load_summary(real_dir)
    if real_summary is None:
        result["status"] = "ERROR"
        result["errors"].append(f"Real summary not found: {real_dir / 'p4_summary.json'}")
        return result

    # Load configs for metadata
    mock_config = load_config(mock_dir) or {}
    real_config = load_config(real_dir) or {}

    # Extract divergence metrics
    mock_div = mock_summary.get("divergence_analysis", {})
    real_div = real_summary.get("divergence_analysis", {})

    mock_divergence_rate = mock_div.get("divergence_rate", 0.0)
    real_divergence_rate = real_div.get("divergence_rate", 0.0)

    # Extract twin accuracy metrics
    mock_accuracy = mock_summary.get("twin_accuracy", {})
    real_accuracy = real_summary.get("twin_accuracy", {})

    mock_twin_success_accuracy = mock_accuracy.get("success_prediction", 0.0)
    real_twin_success_accuracy = real_accuracy.get("success_prediction", 0.0)

    # Compute deltas
    delta_divergence = real_divergence_rate - mock_divergence_rate
    delta_accuracy = real_twin_success_accuracy - mock_twin_success_accuracy

    # Build comparison result
    result["inputs"] = {
        "mock_dir": str(mock_dir),
        "real_dir": str(real_dir),
        "mock_adapter": mock_config.get("telemetry_adapter", "unknown"),
        "real_adapter": real_config.get("telemetry_adapter", "unknown"),
        "mock_cycles": mock_summary.get("execution", {}).get("cycles_completed", 0),
        "real_cycles": real_summary.get("execution", {}).get("cycles_completed", 0),
    }

    result["divergence_comparison"] = {
        "mock_divergence_rate": round(mock_divergence_rate, 4),
        "real_divergence_rate": round(real_divergence_rate, 4),
        "delta_divergence": round(delta_divergence, 4),
        "real_better": delta_divergence < 0,
    }

    result["accuracy_comparison"] = {
        "mock_twin_success_accuracy": round(mock_twin_success_accuracy, 4),
        "real_twin_success_accuracy": round(real_twin_success_accuracy, 4),
        "delta_accuracy": round(delta_accuracy, 4),
        "real_better": delta_accuracy > 0,
    }

    # Extract additional metrics if available
    mock_uplift = mock_summary.get("uplift_metrics", {})
    real_uplift = real_summary.get("uplift_metrics", {})

    result["uplift_comparison"] = {
        "mock_success_rate": round(mock_uplift.get("u2_success_rate_final", 0.0), 4),
        "real_success_rate": round(real_uplift.get("u2_success_rate_final", 0.0), 4),
    }

    # Severity breakdown
    mock_severity = mock_div.get("by_severity", {})
    real_severity = real_div.get("by_severity", {})

    result["severity_breakdown"] = {
        "mock": {
            "minor": mock_severity.get("minor", 0),
            "moderate": mock_severity.get("moderate", 0),
            "severe": mock_severity.get("severe", 0),
        },
        "real": {
            "minor": real_severity.get("minor", 0),
            "moderate": real_severity.get("moderate", 0),
            "severe": real_severity.get("severe", 0),
        },
    }

    # Summary interpretation (SHADOW MODE - advisory only)
    result["interpretation"] = {
        "advisory_only": True,
        "shadow_mode": True,
        "summary": _interpret_comparison(result),
    }

    if verbose:
        print(f"[INFO] Mock divergence rate: {mock_divergence_rate:.2%}", file=sys.stderr)
        print(f"[INFO] Real divergence rate: {real_divergence_rate:.2%}", file=sys.stderr)
        print(f"[INFO] Delta: {delta_divergence:+.2%}", file=sys.stderr)

    return result


def _interpret_comparison(result: Dict[str, Any]) -> str:
    """
    Generate human-readable interpretation of comparison.

    SHADOW MODE: Advisory only, no gating implications.
    """
    div_comp = result.get("divergence_comparison", {})
    mock_div = div_comp.get("mock_divergence_rate", 0)
    real_div = div_comp.get("real_divergence_rate", 0)
    delta = div_comp.get("delta_divergence", 0)

    if real_div < mock_div:
        if delta < -0.10:
            return (
                f"SIGNIFICANT IMPROVEMENT: Real adapter divergence ({real_div:.2%}) "
                f"is {abs(delta):.2%} lower than mock ({mock_div:.2%}). "
                "Twin tracks real telemetry significantly better."
            )
        else:
            return (
                f"IMPROVEMENT: Real adapter divergence ({real_div:.2%}) "
                f"is {abs(delta):.2%} lower than mock ({mock_div:.2%}). "
                "Twin tracks real telemetry better."
            )
    elif real_div > mock_div:
        return (
            f"UNEXPECTED: Real adapter divergence ({real_div:.2%}) "
            f"is {delta:.2%} higher than mock ({mock_div:.2%}). "
            "Investigate adapter configuration or telemetry source."
        )
    else:
        return (
            f"NO DIFFERENCE: Real and mock divergence rates are equal ({real_div:.2%}). "
            "Both adapters produce similar tracking behavior."
        )


def main() -> int:
    """Main entry point."""
    args = parse_args()

    mock_dir = Path(args.mock_dir)
    real_dir = Path(args.real_dir)

    # Validate directories exist
    if not mock_dir.exists():
        print(f"ERROR: Mock directory not found: {mock_dir}", file=sys.stderr)
        return 1

    if not real_dir.exists():
        print(f"ERROR: Real directory not found: {real_dir}", file=sys.stderr)
        return 1

    # Run comparison
    result = compare_runs(mock_dir, real_dir, verbose=args.verbose)

    # Output result
    output_json = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        if args.verbose:
            print(f"[INFO] Comparison written to: {args.output}", file=sys.stderr)
    else:
        print(output_json)

    # SHADOW MODE: Always return 0 (no exit-code gating)
    return 0


if __name__ == "__main__":
    sys.exit(main())
