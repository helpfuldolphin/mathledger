#!/usr/bin/env python3
"""
Veracity Stability Regression Test
Claude A — Veracity Engineer

Verifies that Lean job syntax health remains at PASS-STABLE (0.00% malformation).
Runs dual-scan verification and confirms byte-identical reports.

Usage:
    python tests/test_veracity_stability.py

Exit Codes:
    0 - PASS (veracity maintained)
    1 - FAIL (regression detected)
    2 - ABSTAIN (scan infrastructure unavailable)
"""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run_preflight_scan(output_path: Path) -> tuple[int, str]:
    """
    Run preflight scan and return (exit_code, stdout).
    """
    result = subprocess.run(
        ["python", "tools/preflight_lean_jobs.py", "--json", str(output_path)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    return result.returncode, result.stdout + result.stderr


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of file."""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def load_report(file_path: Path) -> dict:
    """Load and parse JSON report."""
    with open(file_path) as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("VERACITY STABILITY REGRESSION TEST")
    print("Claude A — Veracity Engineer")
    print("=" * 70)
    print()

    # Check if preflight tool exists
    tool_path = Path("tools/preflight_lean_jobs.py")
    if not tool_path.exists():
        print("[ABSTAIN] Preflight tool not found:", tool_path)
        print("Scan infrastructure unavailable")
        return 2

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        scan1_path = tmpdir / "scan1.json"
        scan2_path = tmpdir / "scan2.json"

        # Run first scan
        print("Running scan 1...")
        exit1, output1 = run_preflight_scan(scan1_path)
        print(output1)

        # Run second scan
        print("\nRunning scan 2...")
        exit2, output2 = run_preflight_scan(scan2_path)
        print(output2)

        # Load reports
        report1 = load_report(scan1_path)
        report2 = load_report(scan2_path)

        # Check exit codes
        if exit1 != 0 or exit2 != 0:
            print("\n" + "=" * 70)
            print("[FAIL] Veracity Regression")
            print("=" * 70)
            print(f"Scan 1 exit code: {exit1}")
            print(f"Scan 2 exit code: {exit2}")
            print()
            print(f"Malformation rate: {report1.get('malformation_rate', 'N/A')}")
            print(f"Defects: {report1.get('malformed', 'N/A')}")

            if report1.get('malformed', 0) > 0:
                print("\nDefect Summary:")
                for defect in report1.get('malformed_jobs', []):
                    print(f"  {defect['job_id']}: {defect['pattern']}")
                    print(f"    Hex: {defect.get('hex_window', 'N/A')}")
                    print(f"    Fix: {defect.get('fix_hint', 'N/A')}")

            return 1

        # Check malformation rate
        rate1 = report1.get("malformation_rate", -1)
        rate2 = report2.get("malformation_rate", -1)

        if rate1 != 0.0 or rate2 != 0.0:
            print("\n" + "=" * 70)
            print("[FAIL] Veracity Regression")
            print("=" * 70)
            print(f"Malformation rate: {rate1}")
            print("Expected: 0.00%")
            print()
            print("Defects detected. Veracity compromised.")
            return 1

        # Verify hash stability
        hash1 = compute_sha256(scan1_path)
        hash2 = compute_sha256(scan2_path)

        if hash1 != hash2:
            print("\n" + "=" * 70)
            print("[FAIL] Determinism Violation")
            print("=" * 70)
            print(f"Scan 1 hash: {hash1}")
            print(f"Scan 2 hash: {hash2}")
            print()
            print("Reports not byte-identical. Nondeterminism detected.")
            return 1

        # All checks passed
        print("\n" + "=" * 70)
        print("[PASS] Veracity PASS-STABLE")
        print("=" * 70)
        print(f"Scans: 2")
        print(f"Malformation rate: 0.00%")
        print(f"Jobs: {report1.get('jobs_scanned', 'N/A')}")
        print(f"Valid: {report1.get('valid', 'N/A')} (100.0%)")
        print(f"Malformed: 0")
        print()
        print(f"Hash: {hash1}")
        print("Determinism: VERIFIED (byte-identical)")
        print()
        print("Veracity maintained. No regressions detected.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
