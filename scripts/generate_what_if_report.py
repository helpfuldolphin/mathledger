#!/usr/bin/env python3
"""
Generate What-If Governance Report from First Light Telemetry.

SHADOW MODE ONLY: This script evaluates what governance gates WOULD have done,
but takes NO enforcement action. All output is observational/hypothetical.

Usage:
    python scripts/generate_what_if_report.py \
        --real-cycles-jsonl telemetry.jsonl \
        --output what_if_report.json

Inputs:
    --real-cycles-jsonl: JSONL file from P4/P5 First Light runs
    --output: Output path for what_if_report.json

Each line in the input JSONL should be a JSON object with cycle telemetry.
The script parses fields for G2/G3/G4 gates or derives minimal placeholders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.governance.what_if_engine import (
    WhatIfConfig,
    WhatIfReport,
    build_what_if_report,
    export_what_if_report,
)


# =============================================================================
# TELEMETRY FIELD MAPPING
# =============================================================================

# Field mapping from First Light telemetry to What-If Engine inputs
# Supports multiple naming conventions for flexibility
FIELD_MAP = {
    "cycle": ["cycle", "cycle_num", "step", "iteration"],
    "timestamp": ["timestamp", "ts", "time", "created_at"],
    # G2: Invariant violations
    "invariant_violations": [
        "invariant_violations",
        "violations",
        "invariant_errors",
        "failed_invariants",
    ],
    # G3: Omega (safe region) state
    "in_omega": ["in_omega", "in_safe_region", "is_safe", "omega_safe"],
    "omega_exit_streak": [
        "omega_exit_streak",
        "safe_region_exit_streak",
        "outside_omega_cycles",
        "omega_exit_cycles",
    ],
    # G4: RSI/rho state
    "rho": ["rho", "rsi", "stability_index", "stability"],
    "rho_collapse_streak": [
        "rho_collapse_streak",
        "rsi_streak",
        "stability_collapse_streak",
        "rho_low_streak",
    ],
}


def extract_field(data: Dict[str, Any], field_type: str, default: Any = None) -> Any:
    """
    Extract a field from telemetry data using multiple possible field names.

    Args:
        data: Telemetry dictionary
        field_type: Type of field to extract (key in FIELD_MAP)
        default: Default value if field not found

    Returns:
        Extracted value or default
    """
    possible_names = FIELD_MAP.get(field_type, [field_type])

    for name in possible_names:
        if name in data:
            return data[name]

    return default


def parse_telemetry_line(line: str, line_num: int) -> Optional[Dict[str, Any]]:
    """
    Parse a single JSONL line into telemetry dictionary.

    Args:
        line: JSON line string
        line_num: Line number for error reporting

    Returns:
        Parsed dictionary or None if invalid
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        print(f"Warning: Skipping invalid JSON at line {line_num}: {e}", file=sys.stderr)
        return None


def normalize_telemetry(
    raw_data: Dict[str, Any],
    cycle_num: int,
) -> Dict[str, Any]:
    """
    Normalize telemetry data to What-If Engine input format.

    Extracts G2/G3/G4 relevant fields using field mapping.
    Derives minimal placeholders for missing fields.

    Args:
        raw_data: Raw telemetry dictionary
        cycle_num: Cycle number (used if not in data)

    Returns:
        Normalized telemetry dictionary
    """
    # Extract cycle number (from data or use provided)
    cycle = extract_field(raw_data, "cycle", cycle_num)

    # Extract timestamp
    timestamp = extract_field(
        raw_data,
        "timestamp",
        datetime.now(timezone.utc).isoformat()
    )

    # G2: Invariant violations
    invariant_violations = extract_field(raw_data, "invariant_violations", [])
    if isinstance(invariant_violations, str):
        # Handle comma-separated string format
        invariant_violations = [v.strip() for v in invariant_violations.split(",") if v.strip()]

    # G3: Omega state
    in_omega = extract_field(raw_data, "in_omega", True)
    if isinstance(in_omega, str):
        in_omega = in_omega.lower() in ("true", "1", "yes")

    omega_exit_streak = extract_field(raw_data, "omega_exit_streak", 0)

    # G4: RSI/rho state
    rho = extract_field(raw_data, "rho", 1.0)
    rho_collapse_streak = extract_field(raw_data, "rho_collapse_streak", 0)

    return {
        "cycle": int(cycle),
        "timestamp": str(timestamp),
        "invariant_violations": list(invariant_violations),
        "in_omega": bool(in_omega),
        "omega_exit_streak": int(omega_exit_streak),
        "rho": float(rho),
        "rho_collapse_streak": int(rho_collapse_streak),
    }


def load_telemetry_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load and normalize telemetry from JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of normalized telemetry dictionaries
    """
    telemetry = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            raw_data = parse_telemetry_line(line, line_num)
            if raw_data is None:
                continue

            normalized = normalize_telemetry(raw_data, len(telemetry) + 1)
            telemetry.append(normalized)

    return telemetry


def compute_report_hash(report: WhatIfReport) -> str:
    """
    Compute SHA-256 hash of report for evidence pack attachment.

    Args:
        report: What-If report

    Returns:
        Hex-encoded SHA-256 hash
    """
    report_json = json.dumps(report.to_dict(), sort_keys=True)
    return hashlib.sha256(report_json.encode("utf-8")).hexdigest()


def attach_to_evidence_pack(
    evidence: Dict[str, Any],
    report: WhatIfReport,
) -> Dict[str, Any]:
    """
    Attach What-If report to evidence pack.

    Attaches under: evidence["governance"]["what_if_analysis"]["report"]
    Includes SHA-256 hash for integrity verification.

    Args:
        evidence: Evidence pack dictionary
        report: What-If report to attach

    Returns:
        Updated evidence pack
    """
    # Ensure governance section exists
    if "governance" not in evidence:
        evidence["governance"] = {}

    # Ensure what_if_analysis section exists
    if "what_if_analysis" not in evidence["governance"]:
        evidence["governance"]["what_if_analysis"] = {}

    # Attach report with hash
    report_dict = report.to_dict()
    report_hash = compute_report_hash(report)

    evidence["governance"]["what_if_analysis"]["report"] = report_dict
    evidence["governance"]["what_if_analysis"]["report_sha256"] = report_hash
    evidence["governance"]["what_if_analysis"]["attached_at"] = datetime.now(timezone.utc).isoformat()

    return evidence


def generate_what_if_report(
    telemetry_path: Path,
    output_path: Path,
    run_id: Optional[str] = None,
    config: Optional[WhatIfConfig] = None,
) -> WhatIfReport:
    """
    Generate What-If report from telemetry file.

    Args:
        telemetry_path: Path to input JSONL file
        output_path: Path to output JSON file
        run_id: Optional run identifier
        config: Optional What-If configuration

    Returns:
        Generated WhatIfReport
    """
    # Load telemetry
    telemetry = load_telemetry_jsonl(telemetry_path)

    if not telemetry:
        print(f"Warning: No valid telemetry found in {telemetry_path}", file=sys.stderr)

    # Generate run_id if not provided
    if run_id is None:
        run_id = f"what-if-{telemetry_path.stem}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    # Build report
    report = build_what_if_report(
        telemetry=telemetry,
        config=config,
        run_id=run_id,
    )

    # Export to file
    export_what_if_report(report, str(output_path))

    return report


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate What-If Governance Report from First Light Telemetry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/generate_what_if_report.py \\
        --real-cycles-jsonl results/p5_telemetry.jsonl \\
        --output results/what_if_report.json

    python scripts/generate_what_if_report.py \\
        --real-cycles-jsonl results/first_light.jsonl \\
        --output results/what_if_report.json \\
        --run-id "p5-run-001"

SHADOW MODE: This tool evaluates what gates WOULD have done.
No enforcement action is taken. Mode is always "HYPOTHETICAL".
        """
    )

    parser.add_argument(
        "--real-cycles-jsonl",
        type=Path,
        required=True,
        help="Path to input JSONL file containing cycle telemetry",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output JSON file for What-If report",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier (auto-generated if not provided)",
    )

    # Configuration overrides
    parser.add_argument(
        "--invariant-tolerance",
        type=int,
        default=0,
        help="G2: Number of invariant violations to tolerate (default: 0)",
    )

    parser.add_argument(
        "--omega-exit-threshold",
        type=int,
        default=100,
        help="G3: Cycles outside Omega before triggering (default: 100)",
    )

    parser.add_argument(
        "--rho-min",
        type=float,
        default=0.4,
        help="G4: Minimum rho value threshold (default: 0.4)",
    )

    parser.add_argument(
        "--rho-streak-threshold",
        type=int,
        default=10,
        help="G4: Consecutive cycles below rho-min to trigger (default: 10)",
    )

    parser.add_argument(
        "--attach-evidence",
        type=Path,
        default=None,
        help="Optional path to evidence pack JSON to attach report to",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.real_cycles_jsonl.exists():
        print(f"Error: Input file not found: {args.real_cycles_jsonl}", file=sys.stderr)
        return 1

    # Build configuration
    config = WhatIfConfig(
        invariant_tolerance=args.invariant_tolerance,
        omega_exit_threshold=args.omega_exit_threshold,
        rho_min=args.rho_min,
        rho_streak_threshold=args.rho_streak_threshold,
    )

    # Generate report
    try:
        report = generate_what_if_report(
            telemetry_path=args.real_cycles_jsonl,
            output_path=args.output,
            run_id=args.run_id,
            config=config,
        )
    except Exception as e:
        print(f"Error generating report: {e}", file=sys.stderr)
        return 1

    # Print summary
    print(f"What-If Report Generated: {args.output}")
    print(f"  Mode: {report.mode}")
    print(f"  Run ID: {report.run_id}")
    print(f"  Total Cycles: {report.total_cycles}")
    print(f"  Hypothetical Blocks: {report.hypothetical_blocks}")
    print(f"  Block Rate: {report.hypothetical_block_rate:.2%}")

    if report.blocking_gate_distribution:
        print("  Gate Distribution:")
        for gate, count in report.blocking_gate_distribution.items():
            print(f"    {gate}: {count}")

    # Attach to evidence pack if requested
    if args.attach_evidence:
        if args.attach_evidence.exists():
            with open(args.attach_evidence, "r", encoding="utf-8") as f:
                evidence = json.load(f)
        else:
            evidence = {}

        evidence = attach_to_evidence_pack(evidence, report)

        with open(args.attach_evidence, "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2)

        report_hash = compute_report_hash(report)
        print(f"  Attached to evidence pack: {args.attach_evidence}")
        print(f"  Report SHA-256: {report_hash}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
