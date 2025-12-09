#!/usr/bin/env python3
"""
Docstring Compliance Verifier

PHASE II — NOT RUN IN PHASE I
No uplift claims are made.
Deterministic execution guaranteed.

This script verifies that files implementing metrics, loaders, or runners
include the required Phase II governance markers:
  1. "PHASE II — NOT RUN IN PHASE I"
  2. "No uplift claims are made."
  3. Deterministic execution guarantee

Author: doc-ops-1 (Governance Synchronization Officer)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple


# Required markers
PHASE_II_MARKER = "PHASE II — NOT RUN IN PHASE I"
NO_UPLIFT_MARKER = "No uplift claims are made."
DETERMINISM_KEYWORDS = ["deterministic", "Deterministic", "DETERMINISTIC"]


@dataclass
class ComplianceViolation:
    """A docstring compliance violation."""

    file_path: str
    missing_markers: List[str]
    file_type: str  # metric, loader, runner


def detect_file_type(content: str) -> Set[str]:
    """
    Detect if a file implements metrics, loaders, or runners.

    Args:
        content: File content

    Returns:
        Set of detected types
    """
    types: Set[str] = set()

    # Metric patterns
    metric_patterns = [
        r"def compute_",
        r"def calculate_",
        r"def measure_",
        r"class\s+\w*[Mm]etric",
    ]
    for pattern in metric_patterns:
        if re.search(pattern, content):
            types.add("metric")
            break

    # Loader patterns
    loader_patterns = [
        r"def load_",
        r"class\s+\w*[Ll]oader",
        r"def\s+load\s*\(",
    ]
    for pattern in loader_patterns:
        if re.search(pattern, content):
            types.add("loader")
            break

    # Runner patterns
    runner_patterns = [
        r"def run_",
        r"def execute_",
        r"class\s+\w*[Rr]unner",
    ]
    for pattern in runner_patterns:
        if re.search(pattern, content):
            types.add("runner")
            break

    return types


def check_markers(content: str) -> Tuple[bool, bool, bool]:
    """
    Check for required markers in file content.

    Args:
        content: File content

    Returns:
        Tuple of (has_phase_marker, has_no_uplift, has_determinism)
    """
    has_phase = PHASE_II_MARKER in content
    has_no_uplift = NO_UPLIFT_MARKER in content or "no uplift claim" in content.lower()
    has_determinism = any(kw in content for kw in DETERMINISM_KEYWORDS)

    return has_phase, has_no_uplift, has_determinism


def scan_file(file_path: Path) -> Optional[ComplianceViolation]:
    """
    Scan a single file for compliance violations.

    Args:
        file_path: Path to the file

    Returns:
        ComplianceViolation if issues found, None otherwise
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    file_types = detect_file_type(content)
    if not file_types:
        return None  # Not a metric/loader/runner file

    has_phase, has_no_uplift, has_determinism = check_markers(content)

    missing_markers: List[str] = []
    if not has_phase:
        missing_markers.append(PHASE_II_MARKER)
    if not has_no_uplift:
        missing_markers.append(NO_UPLIFT_MARKER)
    if not has_determinism:
        missing_markers.append("Deterministic execution guarantee")

    if missing_markers:
        return ComplianceViolation(
            file_path=str(file_path),
            missing_markers=missing_markers,
            file_type=", ".join(sorted(file_types)),
        )

    return None


def scan_directory(root_path: Path, scan_dirs: List[str]) -> List[ComplianceViolation]:
    """
    Scan directories for compliance violations.

    Args:
        root_path: Project root path
        scan_dirs: List of directory names to scan

    Returns:
        List of violations
    """
    violations: List[ComplianceViolation] = []

    ignore_dirs = {
        "__pycache__",
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "MagicMock",
    }

    for dir_name in scan_dirs:
        dir_path = root_path / dir_name
        if not dir_path.exists():
            continue

        for py_file in dir_path.rglob("*.py"):
            # Skip ignored directories
            if any(part in ignore_dirs for part in py_file.parts):
                continue

            violation = scan_file(py_file)
            if violation:
                violations.append(violation)

    return violations


def main() -> int:
    """
    Main entry point for docstring compliance verifier.

    Returns:
        Exit code (0 for pass, 1 for violations)
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Docstring Compliance Verifier - Phase II Governance"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Repository root path",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to add missing markers (interactive)",
    )
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="CI mode: exit with error code if violations found",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PHASE II — Docstring Compliance Verifier")
    print("No uplift claims are made. Deterministic execution guaranteed.")
    print("=" * 70)
    print()

    scan_dirs = ["backend", "experiments", "scripts", "rfl"]
    violations = scan_directory(args.root, scan_dirs)

    if not violations:
        print("✓ All metric/loader/runner files are compliant!")
        return 0

    print(f"Found {len(violations)} compliance violation(s):")
    print("-" * 40)

    for v in violations:
        print(f"\nFile: {v.file_path}")
        print(f"Type: {v.file_type}")
        print("Missing markers:")
        for marker in v.missing_markers:
            print(f"  - {marker}")

    if args.fix:
        print("\n" + "-" * 40)
        print("Fix mode: Would add missing markers to files.")
        print("(Not implemented - manual review recommended)")

    if args.ci_mode:
        print(f"\nCI GATE FAILED: {len(violations)} violation(s) found")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

