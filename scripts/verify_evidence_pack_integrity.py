#!/usr/bin/env python3
"""
First-Light Evidence Pack Integrity Verifier

Verifies that all files in an evidence pack match their manifest SHA-256 hashes.

SHADOW MODE CONTRACT:
- This script is purely observational
- It never modifies any files
- It never exits non-zero except on fatal I/O errors
- It is a verification tool, not a gate

Usage:
    python scripts/verify_evidence_pack_integrity.py
    python scripts/verify_evidence_pack_integrity.py --pack-dir results/my_pack
    python scripts/verify_evidence_pack_integrity.py --json-output results/integrity_report.json

Output:
    - Human-readable summary to stdout
    - Optional JSON report to file (--json-output)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FileVerificationResult:
    """Result of verifying a single file."""
    path: str
    expected_hash: str
    status: str = "UNKNOWN"  # "OK", "MISSING", "MISMATCH"
    actual_hash: Optional[str] = None
    expected_size: Optional[int] = None
    actual_size: Optional[int] = None


@dataclass
class IntegrityReport:
    """Complete integrity verification report."""
    schema_version: str = "1.0.0"
    pack_dir: str = ""
    manifest_path: str = ""
    verification_time: str = ""
    status: str = "UNKNOWN"  # "PASSED", "FAILED", "ERROR"
    total_files: int = 0
    ok_count: int = 0
    missing_count: int = 0
    mismatch_count: int = 0
    missing_files: List[str] = field(default_factory=list)
    mismatch_files: List[str] = field(default_factory=list)
    file_results: List[Dict[str, Any]] = field(default_factory=list)
    manifest_hash: str = ""
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "pack_dir": self.pack_dir,
            "manifest_path": self.manifest_path,
            "verification_time": self.verification_time,
            "status": self.status,
            "summary": {
                "total_files": self.total_files,
                "ok_count": self.ok_count,
                "missing_count": self.missing_count,
                "mismatch_count": self.mismatch_count,
            },
            "missing_files": self.missing_files,
            "mismatch_files": self.mismatch_files,
            "file_results": self.file_results,
            "manifest_hash": self.manifest_hash,
            "errors": self.errors,
        }


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def normalize_path(manifest_path: str, pack_dir_name: str) -> str:
    """
    Normalize a manifest path to POSIX format and strip pack directory prefix.

    Handles:
    - Windows backslashes -> forward slashes
    - Paths that include the pack directory name as prefix
    """
    # Convert backslashes to forward slashes
    normalized = manifest_path.replace("\\", "/")

    # Strip pack directory prefix if present
    # e.g., "evidence_pack_first_light/p3_synthetic/file.json" -> "p3_synthetic/file.json"
    parts = normalized.split("/")
    if parts and parts[0] == pack_dir_name:
        normalized = "/".join(parts[1:])
    elif parts and parts[0].startswith("evidence_pack"):
        # Handle variations like "evidence_pack_first_light"
        normalized = "/".join(parts[1:])

    return normalized


def verify_evidence_pack(pack_dir: Path) -> IntegrityReport:
    """
    Verify integrity of an evidence pack against its manifest.

    Args:
        pack_dir: Path to evidence pack directory

    Returns:
        IntegrityReport with verification results
    """
    report = IntegrityReport(
        pack_dir=str(pack_dir),
        verification_time=datetime.now(timezone.utc).isoformat(),
    )

    manifest_path = pack_dir / "manifest.json"
    report.manifest_path = str(manifest_path)

    # Load manifest
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except FileNotFoundError:
        report.status = "ERROR"
        report.errors.append(f"Manifest not found: {manifest_path}")
        return report
    except json.JSONDecodeError as e:
        report.status = "ERROR"
        report.errors.append(f"Invalid manifest JSON: {e}")
        return report

    # Compute manifest hash for tamper detection
    report.manifest_hash = compute_sha256(manifest_path)

    # Get pack directory name for path normalization
    pack_dir_name = pack_dir.name

    # Verify each file
    files = manifest.get("files", [])
    report.total_files = len(files)

    for entry in files:
        raw_path = entry.get("path", "")
        expected_hash = entry.get("sha256", "")
        expected_size = entry.get("size_bytes")

        # Normalize path
        rel_path = normalize_path(raw_path, pack_dir_name)
        file_path = pack_dir / rel_path

        result = FileVerificationResult(
            path=rel_path,
            expected_hash=expected_hash,
            expected_size=expected_size,
        )

        if not file_path.exists():
            result.status = "MISSING"
            report.missing_count += 1
            report.missing_files.append(rel_path)
        else:
            result.actual_hash = compute_sha256(file_path)
            result.actual_size = file_path.stat().st_size

            if result.actual_hash == expected_hash:
                result.status = "OK"
                report.ok_count += 1
            else:
                result.status = "MISMATCH"
                report.mismatch_count += 1
                report.mismatch_files.append(rel_path)

        report.file_results.append({
            "path": result.path,
            "status": result.status,
            "expected_hash": result.expected_hash,
            "actual_hash": result.actual_hash,
            "expected_size": result.expected_size,
            "actual_size": result.actual_size,
        })

    # Determine overall status
    if report.missing_count == 0 and report.mismatch_count == 0:
        report.status = "PASSED"
    else:
        report.status = "FAILED"

    return report


def print_human_readable(report: IntegrityReport) -> None:
    """Print human-readable verification report to stdout."""
    print("=" * 60)
    print("First-Light Evidence Pack Integrity Verification")
    print("=" * 60)
    print()
    print(f"Pack Directory: {report.pack_dir}")
    print(f"Manifest: {report.manifest_path}")
    print(f"Verification Time: {report.verification_time}")
    print()

    if report.errors:
        print("ERRORS:")
        for error in report.errors:
            print(f"  - {error}")
        print()
        print("STATUS: ERROR")
        return

    print("File Verification Results:")
    print("-" * 60)

    for result in report.file_results:
        status = result["status"]
        path = result["path"]

        if status == "OK":
            print(f"  OK: {path}")
        elif status == "MISSING":
            print(f"  MISSING: {path}")
        elif status == "MISMATCH":
            print(f"  MISMATCH: {path}")
            print(f"    Expected: {result['expected_hash']}")
            print(f"    Actual:   {result['actual_hash']}")

    print()
    print("-" * 60)
    print("Summary:")
    print(f"  Total Files: {report.total_files}")
    print(f"  OK: {report.ok_count}")
    print(f"  Missing: {report.missing_count}")
    print(f"  Mismatch: {report.mismatch_count}")
    print()
    print(f"Manifest SHA-256: {report.manifest_hash}")
    print()
    print("=" * 60)
    print(f"INTEGRITY CHECK: {report.status}")
    print("=" * 60)

    if report.status == "FAILED":
        print()
        print("SHADOW MODE NOTE: This is a verification tool, not a gate.")
        print("Failed integrity does not block any operations.")
        print("Investigate the missing/mismatched files above.")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify First-Light Evidence Pack integrity against manifest"
    )
    parser.add_argument(
        "--pack-dir",
        type=str,
        default="results/first_light/evidence_pack_first_light",
        help="Path to evidence pack directory (default: results/first_light/evidence_pack_first_light)",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        help="Path to write JSON report (optional)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable output (use with --json-output)",
    )

    args = parser.parse_args()

    pack_dir = Path(args.pack_dir)

    if not pack_dir.exists():
        print(f"ERROR: Pack directory does not exist: {pack_dir}")
        return 1

    if not pack_dir.is_dir():
        print(f"ERROR: Pack path is not a directory: {pack_dir}")
        return 1

    # Run verification
    report = verify_evidence_pack(pack_dir)

    # Output results
    if not args.quiet:
        print_human_readable(report)

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        if not args.quiet:
            print()
            print(f"JSON report written to: {json_path}")

    # SHADOW MODE: Always exit 0 unless fatal I/O error
    # The report.status indicates PASSED/FAILED for downstream consumption
    return 0


if __name__ == "__main__":
    sys.exit(main())
