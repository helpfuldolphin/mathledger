#!/usr/bin/env python3
"""
Curriculum Drift Check Script

This script checks the Phase II curriculum configuration for drift against
a baseline fingerprint. Intended for CI/CD pipelines and pre-experiment validation.

Usage:
    # Check against default baseline
    python scripts/check_curriculum_drift.py
    
    # Check against custom baseline
    python scripts/check_curriculum_drift.py --baseline path/to/fingerprint.json
    
    # Generate new baseline
    python scripts/check_curriculum_drift.py --generate-baseline path/to/output.json

Exit Codes:
    0: No drift detected (safe to proceed)
    1: Drift detected (review changes before proceeding)
    2: Configuration error (invalid curriculum or missing baseline)
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.curriculum_loader_v2 import (
    CurriculumFingerprint,
    CurriculumLoaderV2,
    CurriculumValidationError,
    check_drift,
    compute_curriculum_fingerprint,
)


def find_default_baseline() -> Path:
    """Find the default baseline fingerprint file."""
    # Try experiments/prereg/ first
    candidates = [
        Path("experiments/prereg/curriculum_fingerprint_u2.json"),
        Path("experiments/prereg/curriculum_baseline.json"),
        Path("baseline_fingerprint.json"),
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    raise FileNotFoundError(
        "No baseline fingerprint found. Expected one of:\n" +
        "\n".join(f"  - {c}" for c in candidates)
    )


def load_baseline(baseline_path: Path) -> CurriculumFingerprint:
    """Load baseline fingerprint from JSON file."""
    try:
        with open(baseline_path, 'r') as f:
            data = json.load(f)
        return CurriculumFingerprint.from_dict(data)
    except FileNotFoundError:
        print(f"Error: Baseline file not found: {baseline_path}", file=sys.stderr)
        sys.exit(2)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error: Invalid baseline file: {e}", file=sys.stderr)
        sys.exit(2)


def generate_baseline(loader: CurriculumLoaderV2, output_path: Path) -> None:
    """Generate and save baseline fingerprint."""
    fingerprint = compute_curriculum_fingerprint(loader)
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(fingerprint.to_dict(), f, indent=2)
    
    print(f"✓ Baseline fingerprint saved to {output_path}")
    print(f"  Schema: {fingerprint.schema_version}")
    print(f"  Slices: {fingerprint.slice_count}")
    print(f"  Hash: {fingerprint.hash[:16]}...")


def check_for_drift(
    loader: CurriculumLoaderV2,
    baseline_path: Path,
    verbose: bool = False,
) -> int:
    """Check for curriculum drift. Returns exit code."""
    # Load baseline
    expected = load_baseline(baseline_path)
    
    # Compute current fingerprint
    current = compute_curriculum_fingerprint(loader)
    
    # Check for drift
    report = check_drift(current, expected)
    
    if report.matches:
        print("✓ No drift detected — curriculum matches baseline")
        if verbose:
            print(f"  Baseline: {baseline_path}")
            print(f"  Schema: {current.schema_version}")
            print(f"  Slices: {current.slice_count}")
            print(f"  Hash: {current.hash[:16]}...")
        return 0
    else:
        print("✗ Drift detected — curriculum differs from baseline", file=sys.stderr)
        print(f"  Baseline: {baseline_path}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Differences:", file=sys.stderr)
        for diff in report.differences:
            print(f"  - {diff}", file=sys.stderr)
        print("", file=sys.stderr)
        print("If these changes are intentional:", file=sys.stderr)
        print(f"  1. Review the changes carefully", file=sys.stderr)
        print(f"  2. Update the baseline:", file=sys.stderr)
        print(f"     python -m experiments.curriculum_loader_v2 --fingerprint --json > {baseline_path}", file=sys.stderr)
        print(f"  3. Commit the updated baseline", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Check Phase II curriculum for drift against baseline fingerprint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        '--baseline',
        type=Path,
        help='Path to baseline fingerprint JSON (default: auto-discover)',
    )
    
    parser.add_argument(
        '--generate-baseline',
        type=Path,
        metavar='OUTPUT',
        help='Generate and save baseline fingerprint to OUTPUT',
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to curriculum YAML (default: config/curriculum_uplift_phase2.yaml)',
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output',
    )
    
    args = parser.parse_args()
    
    # Load curriculum
    try:
        if args.config:
            loader = CurriculumLoaderV2.from_yaml(args.config)
        else:
            loader = CurriculumLoaderV2.from_default_phase2_config()
        
        if args.verbose:
            print(f"✓ Curriculum loaded successfully")
            print(f"  Schema: {loader.schema_version}")
            print(f"  Slices: {len(loader.slices)}")
            print("")
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except CurriculumValidationError as e:
        print(f"Error: Curriculum validation failed:\n{e}", file=sys.stderr)
        sys.exit(2)
    
    # Generate baseline mode
    if args.generate_baseline:
        generate_baseline(loader, args.generate_baseline)
        sys.exit(0)
    
    # Drift check mode
    baseline_path = args.baseline
    if baseline_path is None:
        try:
            baseline_path = find_default_baseline()
            if args.verbose:
                print(f"Using baseline: {baseline_path}")
                print("")
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("", file=sys.stderr)
            print("To generate a baseline:", file=sys.stderr)
            print(f"  python {sys.argv[0]} --generate-baseline baseline.json", file=sys.stderr)
            sys.exit(2)
    
    exit_code = check_for_drift(loader, baseline_path, verbose=args.verbose)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
