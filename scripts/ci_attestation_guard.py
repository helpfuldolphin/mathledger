#!/usr/bin/env python3
"""
CI Attestation Envelope Guard
==============================

Exit codes:
- 0: PASS - All attestation chains valid
- 1: PARTIAL - Non-critical warnings found (hash drift, schema drift, timestamp issues)
- 2: FAIL - Critical issues found (chain discontinuities, duplicates, dual-root mismatches)
- 3: CRITICAL - System-level failure (block entire repo)

Usage:
    python scripts/ci_attestation_guard.py [artifacts_dir] [--strict-schema] [--fail-on-warnings]

Examples:
    # Basic verification
    python scripts/ci_attestation_guard.py artifacts/

    # Strict schema checking
    python scripts/ci_attestation_guard.py artifacts/ --strict-schema

    # Fail on warnings
    python scripts/ci_attestation_guard.py artifacts/ --fail-on-warnings
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import attestation module
sys.path.insert(0, str(Path(__file__).parent.parent))

from attestation.cross_chain_verifier import CrossChainVerifier


# Exit codes
EXIT_PASS = 0
EXIT_PARTIAL = 1
EXIT_FAIL = 2
EXIT_CRITICAL = 3


def main() -> int:
    """Main entry point for CI attestation guard."""
    parser = argparse.ArgumentParser(
        description='CI Attestation Envelope Guard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'artifacts_dir',
        type=Path,
        nargs='?',
        default=Path('artifacts'),
        help='Path to artifacts directory (default: artifacts/)',
    )
    parser.add_argument(
        '--strict-schema',
        action='store_true',
        help='Enable strict schema checking (flag extra fields)',
    )
    parser.add_argument(
        '--fail-on-warnings',
        action='store_true',
        help='Fail (exit 2) on warnings instead of partial (exit 1)',
    )
    parser.add_argument(
        '--manifest-pattern',
        default='**/attestation.json',
        help='Glob pattern for manifest files (default: **/attestation.json)',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output (only print final status)',
    )
    
    args = parser.parse_args()
    
    # Check if artifacts directory exists
    if not args.artifacts_dir.exists():
        if not args.quiet:
            print(f"ERROR: Artifacts directory not found: {args.artifacts_dir}", file=sys.stderr)
            print("This may indicate experiments have not been run yet.", file=sys.stderr)
        # No artifacts = CRITICAL (prevents broken builds)
        return EXIT_CRITICAL
    
    if not args.artifacts_dir.is_dir():
        if not args.quiet:
            print(f"ERROR: Path is not a directory: {args.artifacts_dir}", file=sys.stderr)
        return EXIT_CRITICAL
    
    # Create verifier
    verifier = CrossChainVerifier(strict_schema=args.strict_schema)
    
    # Verify attestation chain
    try:
        result = verifier.verify_artifacts_directory(
            args.artifacts_dir,
            manifest_pattern=args.manifest_pattern,
        )
    except Exception as e:
        if not args.quiet:
            print(f"CRITICAL ERROR during verification: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        return EXIT_CRITICAL
    
    # Print summary
    if not args.quiet:
        print(result.summary())
        print()
    
    # Determine exit code
    if not result.is_valid:
        # Critical issues found
        if not args.quiet:
            print("❌ FAIL: Critical attestation issues detected", file=sys.stderr)
        return EXIT_FAIL
    
    if result.has_warnings:
        # Non-critical warnings
        if args.fail_on_warnings:
            if not args.quiet:
                print("❌ FAIL: Warnings detected (--fail-on-warnings enabled)", file=sys.stderr)
            return EXIT_FAIL
        else:
            if not args.quiet:
                print("⚠️  PARTIAL: Non-critical warnings detected", file=sys.stderr)
            return EXIT_PARTIAL
    
    # All clear
    if not args.quiet:
        print("✅ PASS: All attestation chains verified successfully")
    return EXIT_PASS


if __name__ == '__main__':
    sys.exit(main())
