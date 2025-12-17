#!/usr/bin/env python3
"""
Toolchain Parity Verification Script

Verifies that local toolchain matches the committed baseline.
Used in CI to ensure all environments use identical toolchains.

SAVE TO REPO: YES
Rationale: CI parity enforcement. Ensures local == CI toolchain.

Usage:
    python scripts/verify_toolchain_parity.py
    python scripts/verify_toolchain_parity.py --baseline custom_baseline.json
    python scripts/verify_toolchain_parity.py --update  # Update baseline
"""

import argparse
import json
import sys
from pathlib import Path

# Add repo root to path for imports
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

from substrate.repro.toolchain import (
    capture_toolchain_snapshot,
    load_toolchain_snapshot,
    save_toolchain_snapshot,
    verify_toolchain_match,
)


DEFAULT_BASELINE = REPO_ROOT / "toolchain_baseline.json"


def verify_parity(baseline_path: Path, verbose: bool = False) -> bool:
    """
    Verify local toolchain matches baseline.

    Returns True if match, False otherwise.
    """
    if not baseline_path.exists():
        print(f"ERROR: Baseline not found: {baseline_path}")
        print("Run with --update to create baseline")
        return False

    if verbose:
        print(f"Loading baseline from {baseline_path}")

    baseline = load_toolchain_snapshot(baseline_path)
    current = capture_toolchain_snapshot(REPO_ROOT)

    if verbose:
        print(f"Baseline fingerprint: {baseline.fingerprint[:16]}...")
        print(f"Current fingerprint:  {current.fingerprint[:16]}...")

    match, differences = verify_toolchain_match(baseline, current, strict=True)

    if match:
        print("PASS: Toolchain matches baseline")
        print(f"  Fingerprint: {current.fingerprint}")
        return True
    else:
        print("FAIL: Toolchain mismatch detected")
        print()
        print("Differences:")
        for diff in differences:
            print(f"  - {diff}")
        print()
        print("Expected (baseline):")
        print(f"  Python:     {baseline.python.version}")
        print(f"  uv:         {baseline.python.uv_version}")
        print(f"  Lean:       {baseline.lean.version}")
        print(f"  Fingerprint: {baseline.fingerprint}")
        print()
        print("Actual (current):")
        print(f"  Python:     {current.python.version}")
        print(f"  uv:         {current.python.uv_version}")
        print(f"  Lean:       {current.lean.version}")
        print(f"  Fingerprint: {current.fingerprint}")
        print()
        print("To update baseline, run: python scripts/verify_toolchain_parity.py --update")
        return False


def update_baseline(baseline_path: Path) -> None:
    """Update baseline with current toolchain."""
    current = capture_toolchain_snapshot(REPO_ROOT)
    save_toolchain_snapshot(current, baseline_path)
    print(f"Baseline updated: {baseline_path}")
    print(f"  Fingerprint: {current.fingerprint}")
    print(f"  Python:      {current.python.version}")
    print(f"  uv:          {current.python.uv_version}")
    print(f"  Lean:        {current.lean.version}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify toolchain parity with baseline"
    )
    parser.add_argument(
        "--baseline", "-b",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"Baseline file (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--update", "-u",
        action="store_true",
        help="Update baseline with current toolchain",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    if args.update:
        update_baseline(args.baseline)
        sys.exit(0)

    success = verify_parity(args.baseline, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
