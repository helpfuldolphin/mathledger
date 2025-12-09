#!/usr/bin/env python3
# PHASE II — NOT USED IN PHASE I
"""
PRNG Manifest Diff Tool — Compare Determinism Between Experiment Runs.

This tool compares two manifests' PRNG schedules and attestation blocks
to identify how determinism differs between runs.

Comparison Status:
    EQUIVALENT   - Manifests have identical PRNG configurations
    DRIFTED      - Same master seed but different derived values
    INCOMPATIBLE - Different derivation schemes or incompatible configurations

Exit Codes:
    0 - EQUIVALENT (manifests match)
    1 - DRIFTED (same seed, different derivation)
    2 - INCOMPATIBLE (fundamentally different configurations)
    3 - Error during comparison

Usage:
    python scripts/prng_manifest_diff.py manifest1.json manifest2.json
    python scripts/prng_manifest_diff.py manifest1.json manifest2.json --json
    python scripts/prng_manifest_diff.py manifest1.json manifest2.json --verbose

Author: Agent A2 (runtime-ops-2)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DiffStatus(str, Enum):
    """Status of manifest comparison."""
    EQUIVALENT = "EQUIVALENT"
    DRIFTED = "DRIFTED"
    INCOMPATIBLE = "INCOMPATIBLE"
    ERROR = "ERROR"


class DiffSeverity(str, Enum):
    """Severity of a difference."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Difference:
    """A single difference between manifests."""
    field: str
    left_value: Any
    right_value: Any
    severity: DiffSeverity
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "left": self.left_value,
            "right": self.right_value,
            "severity": self.severity.value,
            "message": self.message,
        }


@dataclass
class DiffResult:
    """Result of manifest comparison."""
    status: DiffStatus
    left_path: str
    right_path: str
    differences: List[Difference]
    summary: str
    left_attestation: Optional[Dict[str, Any]] = None
    right_attestation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "left_path": self.left_path,
            "right_path": self.right_path,
            "summary": self.summary,
            "differences": [d.to_dict() for d in self.differences],
            "left_attestation": self.left_attestation,
            "right_attestation": self.right_attestation,
        }


def load_manifest(path: Path) -> Dict[str, Any]:
    """Load and validate a manifest file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_attestation(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract PRNG attestation from manifest."""
    return manifest.get('prng_attestation') or manifest.get('prng')


def compare_field(
    field_name: str,
    left: Any,
    right: Any,
    severity: DiffSeverity = DiffSeverity.WARNING,
    format_fn: Optional[callable] = None,
) -> Optional[Difference]:
    """Compare a single field and return difference if any."""
    if left == right:
        return None

    left_str = format_fn(left) if format_fn else str(left)
    right_str = format_fn(right) if format_fn else str(right)

    return Difference(
        field=field_name,
        left_value=left_str,
        right_value=right_str,
        severity=severity,
        message=f"{field_name} differs",
    )


def truncate_hex(value: Optional[str], length: int = 16) -> str:
    """Truncate a hex string for display."""
    if value is None:
        return "<none>"
    if len(value) <= length:
        return value
    return f"{value[:length]}..."


def compare_manifests(
    left_path: Path,
    right_path: Path,
    verbose: bool = False,
) -> DiffResult:
    """
    Compare two manifests' PRNG configurations.

    Args:
        left_path: Path to first manifest.
        right_path: Path to second manifest.
        verbose: Include additional details.

    Returns:
        DiffResult with comparison status and details.
    """
    differences: List[Difference] = []

    try:
        left_manifest = load_manifest(left_path)
        right_manifest = load_manifest(right_path)
    except FileNotFoundError as e:
        return DiffResult(
            status=DiffStatus.ERROR,
            left_path=str(left_path),
            right_path=str(right_path),
            differences=[],
            summary=f"File not found: {e.filename}",
        )
    except json.JSONDecodeError as e:
        return DiffResult(
            status=DiffStatus.ERROR,
            left_path=str(left_path),
            right_path=str(right_path),
            differences=[],
            summary=f"Invalid JSON: {e}",
        )

    left_attest = extract_attestation(left_manifest)
    right_attest = extract_attestation(right_manifest)

    # Check for missing attestation
    if left_attest is None and right_attest is None:
        return DiffResult(
            status=DiffStatus.EQUIVALENT,
            left_path=str(left_path),
            right_path=str(right_path),
            differences=[],
            summary="Both manifests lack PRNG attestation (legacy format)",
        )

    if left_attest is None or right_attest is None:
        differences.append(Difference(
            field="prng_attestation",
            left_value="present" if left_attest else "missing",
            right_value="present" if right_attest else "missing",
            severity=DiffSeverity.ERROR,
            message="One manifest lacks PRNG attestation",
        ))
        return DiffResult(
            status=DiffStatus.INCOMPATIBLE,
            left_path=str(left_path),
            right_path=str(right_path),
            differences=differences,
            summary="PRNG attestation present in only one manifest",
            left_attestation=left_attest,
            right_attestation=right_attest,
        )

    # Compare schema version
    left_schema = left_attest.get('schema_version', '1.0')
    right_schema = right_attest.get('schema_version', '1.0')
    if diff := compare_field('schema_version', left_schema, right_schema, DiffSeverity.WARNING):
        differences.append(diff)

    # Compare derivation scheme (INCOMPATIBLE if different)
    left_scheme = left_attest.get('derivation_scheme')
    right_scheme = right_attest.get('derivation_scheme')
    if left_scheme != right_scheme:
        differences.append(Difference(
            field="derivation_scheme",
            left_value=left_scheme,
            right_value=right_scheme,
            severity=DiffSeverity.ERROR,
            message="Derivation schemes are incompatible",
        ))
        return DiffResult(
            status=DiffStatus.INCOMPATIBLE,
            left_path=str(left_path),
            right_path=str(right_path),
            differences=differences,
            summary="Different derivation schemes make comparison invalid",
            left_attestation=left_attest,
            right_attestation=right_attest,
        )

    # Compare master seed
    left_seed = left_attest.get('master_seed_hex')
    right_seed = right_attest.get('master_seed_hex')
    seeds_match = left_seed == right_seed

    if not seeds_match:
        differences.append(Difference(
            field="master_seed_hex",
            left_value=truncate_hex(left_seed),
            right_value=truncate_hex(right_seed),
            severity=DiffSeverity.INFO,
            message="Master seeds differ (expected for different experiments)",
        ))

    # Compare implementation
    left_impl = left_attest.get('implementation')
    right_impl = right_attest.get('implementation')
    if diff := compare_field('implementation', left_impl, right_impl, DiffSeverity.INFO):
        differences.append(diff)

    # Compare lineage entry count
    left_count = left_attest.get('lineage_entry_count')
    right_count = right_attest.get('lineage_entry_count')
    if diff := compare_field('lineage_entry_count', left_count, right_count, DiffSeverity.WARNING):
        differences.append(diff)

    # Compare lineage Merkle root
    left_merkle = left_attest.get('lineage_merkle_root')
    right_merkle = right_attest.get('lineage_merkle_root')

    if left_merkle and right_merkle and left_merkle != right_merkle:
        differences.append(Difference(
            field="lineage_merkle_root",
            left_value=truncate_hex(left_merkle),
            right_value=truncate_hex(right_merkle),
            severity=DiffSeverity.WARNING if not seeds_match else DiffSeverity.ERROR,
            message="Lineage Merkle roots differ" + (
                " (DRIFT: same seed, different derivation)" if seeds_match else ""
            ),
        ))

    # Compare integration tests status
    left_tests = left_attest.get('integration_tests_passed')
    right_tests = right_attest.get('integration_tests_passed')
    if diff := compare_field('integration_tests_passed', left_tests, right_tests, DiffSeverity.INFO):
        differences.append(diff)

    # Compare experiment configuration if available
    left_config = left_manifest.get('configuration', {}).get('snapshot', {})
    right_config = right_manifest.get('configuration', {}).get('snapshot', {})

    if verbose:
        # Add config differences
        for key in set(left_config.keys()) | set(right_config.keys()):
            if key in ('random_seed',):  # Skip seed itself
                continue
            left_val = left_config.get(key)
            right_val = right_config.get(key)
            if left_val != right_val:
                differences.append(Difference(
                    field=f"config.{key}",
                    left_value=left_val,
                    right_value=right_val,
                    severity=DiffSeverity.INFO,
                    message=f"Configuration '{key}' differs",
                ))

    # Determine overall status
    error_diffs = [d for d in differences if d.severity == DiffSeverity.ERROR]
    warning_diffs = [d for d in differences if d.severity == DiffSeverity.WARNING]

    if error_diffs:
        if seeds_match:
            status = DiffStatus.DRIFTED
            summary = "Same master seed but derivation drifted"
        else:
            status = DiffStatus.INCOMPATIBLE
            summary = "Manifests are incompatible for comparison"
    elif warning_diffs:
        if seeds_match:
            status = DiffStatus.DRIFTED
            summary = "Same master seed with minor differences"
        else:
            status = DiffStatus.EQUIVALENT
            summary = "Different experiments, no unexpected drift"
    elif differences:
        status = DiffStatus.EQUIVALENT
        summary = "Manifests are equivalent (info-level differences only)"
    else:
        status = DiffStatus.EQUIVALENT
        summary = "Manifests are identical"

    return DiffResult(
        status=status,
        left_path=str(left_path),
        right_path=str(right_path),
        differences=differences,
        summary=summary,
        left_attestation=left_attest,
        right_attestation=right_attest,
    )


def format_human_output(result: DiffResult) -> str:
    """Format result for human-readable output."""
    lines = []
    lines.append("=" * 70)
    lines.append("PRNG Manifest Diff — Agent A2 (runtime-ops-2)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Left:  {result.left_path}")
    lines.append(f"Right: {result.right_path}")
    lines.append("")

    # Status with icon
    status_icons = {
        DiffStatus.EQUIVALENT: "✅",
        DiffStatus.DRIFTED: "⚠️",
        DiffStatus.INCOMPATIBLE: "❌",
        DiffStatus.ERROR: "💥",
    }
    icon = status_icons.get(result.status, "?")
    lines.append(f"Status: {icon} {result.status.value}")
    lines.append(f"Summary: {result.summary}")
    lines.append("")

    if result.differences:
        lines.append("-" * 70)
        lines.append("Differences:")
        lines.append("-" * 70)

        # Group by severity
        for severity in [DiffSeverity.ERROR, DiffSeverity.WARNING, DiffSeverity.INFO]:
            severity_diffs = [d for d in result.differences if d.severity == severity]
            if not severity_diffs:
                continue

            severity_icons = {
                DiffSeverity.ERROR: "❌",
                DiffSeverity.WARNING: "⚠️",
                DiffSeverity.INFO: "ℹ️",
            }
            lines.append(f"\n{severity_icons[severity]} {severity.value.upper()}:")

            for diff in severity_diffs:
                lines.append(f"  • {diff.field}")
                lines.append(f"      Left:  {diff.left_value}")
                lines.append(f"      Right: {diff.right_value}")

        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PRNG Manifest Diff — Compare Determinism Between Runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
    0 - EQUIVALENT (manifests match)
    1 - DRIFTED (same seed, different derivation)
    2 - INCOMPATIBLE (fundamentally different configurations)
    3 - Error during comparison

Examples:
    python scripts/prng_manifest_diff.py run1/manifest.json run2/manifest.json
    python scripts/prng_manifest_diff.py manifest1.json manifest2.json --json
    python scripts/prng_manifest_diff.py manifest1.json manifest2.json --verbose
        """,
    )
    parser.add_argument(
        "left",
        type=Path,
        help="Path to first manifest",
    )
    parser.add_argument(
        "right",
        type=Path,
        help="Path to second manifest",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output JSON format",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Include configuration differences",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Write output to file",
    )

    args = parser.parse_args()

    result = compare_manifests(args.left, args.right, args.verbose)

    if args.json:
        output = json.dumps(result.to_dict(), indent=2)
    else:
        output = format_human_output(result)

    if args.output:
        args.output.write_text(output)
        print(f"Output written to: {args.output}")
    else:
        print(output)

    # Return exit code based on status
    exit_codes = {
        DiffStatus.EQUIVALENT: 0,
        DiffStatus.DRIFTED: 1,
        DiffStatus.INCOMPATIBLE: 2,
        DiffStatus.ERROR: 3,
    }
    return exit_codes.get(result.status, 3)


if __name__ == "__main__":
    sys.exit(main())

