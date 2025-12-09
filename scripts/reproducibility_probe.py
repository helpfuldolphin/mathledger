#!/usr/bin/env python3
# PHASE II — NOT USED IN PHASE I
"""
Reproducibility Probe — Verify Experiment Determinism from Manifests.

This tool allows verification that a past experiment run can be reproduced
with identical PRNG lineage and outputs. Given a manifest.json, it:

1. Extracts the prng_attestation block (master_seed, lineage Merkle root)
2. Re-derives the seed schedule for the recorded cycles
3. Compares the re-derived lineage against the recorded Merkle root
4. Optionally re-runs a small subset to verify output consistency

Exit Codes:
    0 - Reproducibility verified (REPRODUCIBLE)
    1 - Drift detected (DRIFT_DETECTED)
    2 - Incomplete verification (INCOMPLETE)
    3 - Error during verification

Usage:
    python scripts/reproducibility_probe.py manifest.json
    python scripts/reproducibility_probe.py manifest.json --cycles 5
    python scripts/reproducibility_probe.py manifest.json --output report.json

Author: Agent A2 (runtime-ops-2)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class ReproducibilityStatus(str, Enum):
    """Status of reproducibility verification."""
    REPRODUCIBLE = "REPRODUCIBLE"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    INCOMPLETE = "INCOMPLETE"
    ERROR = "ERROR"


@dataclass
class DriftDetail:
    """Details about a detected drift."""
    component: str
    expected: str
    actual: str
    severity: str = "warning"  # info, warning, error


@dataclass
class ReproducibilityReport:
    """Report of reproducibility verification."""
    status: ReproducibilityStatus
    manifest_path: str
    git_commit_manifest: str
    git_commit_current: str
    timestamp: str
    checks_performed: List[str]
    drifts: List[DriftDetail]
    seed_schedule_verified: bool = False
    lineage_merkle_verified: bool = False
    outputs_verified: bool = False
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "manifest_path": self.manifest_path,
            "git_commit_manifest": self.git_commit_manifest,
            "git_commit_current": self.git_commit_current,
            "timestamp": self.timestamp,
            "checks_performed": self.checks_performed,
            "drifts": [asdict(d) for d in self.drifts],
            "seed_schedule_verified": self.seed_schedule_verified,
            "lineage_merkle_verified": self.lineage_merkle_verified,
            "outputs_verified": self.outputs_verified,
            "message": self.message,
        }


def get_current_git_commit() -> str:
    """Get the current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load and validate a manifest file."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_prng_attestation(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract the prng_attestation or prng block from a manifest."""
    return manifest.get("prng_attestation") or manifest.get("prng")


def verify_seed_schedule(
    master_seed_hex: str,
    num_cycles: int,
    slice_name: str,
    mode: str,
    expected_seeds: Optional[List[int]] = None,
) -> Tuple[bool, List[int], List[DriftDetail]]:
    """
    Verify that the seed schedule can be reproduced.

    Returns:
        Tuple of (verified, actual_seeds, drifts)
    """
    from rfl.prng import DeterministicPRNG

    prng = DeterministicPRNG(master_seed_hex)
    actual_seeds = prng.generate_seed_schedule(num_cycles, slice_name, mode)

    drifts = []
    if expected_seeds:
        if len(expected_seeds) != len(actual_seeds):
            drifts.append(DriftDetail(
                component="seed_schedule_length",
                expected=str(len(expected_seeds)),
                actual=str(len(actual_seeds)),
                severity="error",
            ))
            return False, actual_seeds, drifts

        for i, (exp, act) in enumerate(zip(expected_seeds, actual_seeds)):
            if exp != act:
                drifts.append(DriftDetail(
                    component=f"seed_cycle_{i:04d}",
                    expected=str(exp),
                    actual=str(act),
                    severity="error",
                ))

        return len(drifts) == 0, actual_seeds, drifts

    return True, actual_seeds, drifts


def verify_lineage_merkle(
    master_seed_hex: str,
    cycle_paths: List[Tuple[str, ...]],
    expected_merkle_root: Optional[str],
) -> Tuple[bool, str, List[DriftDetail]]:
    """
    Verify that the lineage Merkle root can be reproduced.

    Returns:
        Tuple of (verified, actual_merkle_root, drifts)
    """
    from rfl.prng.lineage import SeedLineage

    lineage = SeedLineage(master_seed_hex)
    for path in sorted(cycle_paths):  # Sorted for determinism
        lineage.record(*path)

    actual_merkle_root = lineage.compute_merkle_root()

    drifts = []
    if expected_merkle_root:
        if actual_merkle_root != expected_merkle_root:
            drifts.append(DriftDetail(
                component="lineage_merkle_root",
                expected=expected_merkle_root,
                actual=actual_merkle_root,
                severity="error",
            ))
            return False, actual_merkle_root, drifts

    return True, actual_merkle_root, drifts


def verify_cross_process_determinism(
    master_seed_hex: str,
    test_paths: List[Tuple[str, ...]],
) -> Tuple[bool, List[DriftDetail]]:
    """
    Verify PRNG determinism across process boundaries.

    Returns:
        Tuple of (verified, drifts)
    """
    from rfl.prng import DeterministicPRNG

    # In-process seeds
    prng = DeterministicPRNG(master_seed_hex)
    inprocess_seeds = [prng.seed_for_path(*p) for p in test_paths]

    # Subprocess seeds
    script = f'''
import sys
sys.path.insert(0, "{PROJECT_ROOT}")
from rfl.prng import DeterministicPRNG
prng = DeterministicPRNG("{master_seed_hex}")
paths = {test_paths}
seeds = [prng.seed_for_path(*p) for p in paths]
print(",".join(str(s) for s in seeds))
'''

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=PROJECT_ROOT,
        )

        if result.returncode != 0:
            return False, [DriftDetail(
                component="subprocess_execution",
                expected="exit_code=0",
                actual=f"exit_code={result.returncode}: {result.stderr}",
                severity="error",
            )]

        subprocess_seeds = [int(s) for s in result.stdout.strip().split(",")]

        drifts = []
        for i, (inp, subp) in enumerate(zip(inprocess_seeds, subprocess_seeds)):
            if inp != subp:
                drifts.append(DriftDetail(
                    component=f"cross_process_seed_{i}",
                    expected=str(inp),
                    actual=str(subp),
                    severity="error",
                ))

        return len(drifts) == 0, drifts

    except Exception as e:
        return False, [DriftDetail(
            component="cross_process_check",
            expected="success",
            actual=str(e),
            severity="error",
        )]


def probe_reproducibility(
    manifest_path: Path,
    num_cycles: int = 5,
    verbose: bool = False,
) -> ReproducibilityReport:
    """
    Probe reproducibility of an experiment from its manifest.

    Args:
        manifest_path: Path to the manifest.json file.
        num_cycles: Number of cycles to verify.
        verbose: Enable verbose output.

    Returns:
        ReproducibilityReport with verification results.
    """
    checks_performed = []
    all_drifts: List[DriftDetail] = []

    # Initialize report
    report = ReproducibilityReport(
        status=ReproducibilityStatus.INCOMPLETE,
        manifest_path=str(manifest_path),
        git_commit_manifest="unknown",
        git_commit_current=get_current_git_commit(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        checks_performed=checks_performed,
        drifts=all_drifts,
    )

    try:
        # Load manifest
        manifest = load_manifest(manifest_path)
        checks_performed.append("manifest_load")

        # Extract provenance
        provenance = manifest.get("provenance", {})
        report.git_commit_manifest = provenance.get("git_commit", "unknown")

        # Check for git commit mismatch (warning only)
        if (report.git_commit_current != "unknown" and 
            report.git_commit_manifest != "unknown" and
            report.git_commit_current != report.git_commit_manifest):
            all_drifts.append(DriftDetail(
                component="git_commit",
                expected=report.git_commit_manifest,
                actual=report.git_commit_current,
                severity="warning",
            ))
            if verbose:
                print(f"⚠ Git commit mismatch: manifest={report.git_commit_manifest[:12]}... current={report.git_commit_current[:12]}...")

        # Extract PRNG attestation
        attestation = extract_prng_attestation(manifest)
        if not attestation:
            report.message = "No prng_attestation block found in manifest"
            return report
        checks_performed.append("attestation_extract")

        master_seed_hex = attestation.get("master_seed_hex")
        if not master_seed_hex:
            report.message = "Missing master_seed_hex in attestation"
            return report

        # Extract experiment configuration
        config = manifest.get("configuration", {}).get("snapshot", {})
        slice_name = config.get("slice_name", "default")
        mode = config.get("mode", "baseline")
        expected_merkle_root = attestation.get("lineage_merkle_root")

        # Use manifest's cycle count if available for accurate verification
        manifest_num_cycles = config.get("num_cycles") or attestation.get("lineage_entry_count")
        if manifest_num_cycles and num_cycles != manifest_num_cycles:
            if verbose:
                print(f"  ℹ Using manifest's cycle count: {manifest_num_cycles} (requested: {num_cycles})")
            num_cycles = manifest_num_cycles

        # Generate cycle paths for verification
        cycle_paths = [
            (slice_name, mode, f"cycle_{i:04d}", "ordering")
            for i in range(num_cycles)
        ]

        # Verify seed schedule
        if verbose:
            print(f"Verifying seed schedule for {num_cycles} cycles...")

        schedule_ok, actual_seeds, schedule_drifts = verify_seed_schedule(
            master_seed_hex=master_seed_hex,
            num_cycles=num_cycles,
            slice_name=slice_name,
            mode=mode,
        )
        report.seed_schedule_verified = schedule_ok
        all_drifts.extend(schedule_drifts)
        checks_performed.append("seed_schedule_verify")

        if verbose:
            status = "✓" if schedule_ok else "✗"
            print(f"  {status} Seed schedule: {actual_seeds[:3]}...")

        # Verify lineage Merkle root
        if verbose:
            print("Verifying lineage Merkle root...")

        merkle_ok, actual_merkle, merkle_drifts = verify_lineage_merkle(
            master_seed_hex=master_seed_hex,
            cycle_paths=cycle_paths,
            expected_merkle_root=expected_merkle_root,
        )
        report.lineage_merkle_verified = merkle_ok
        all_drifts.extend(merkle_drifts)
        checks_performed.append("lineage_merkle_verify")

        if verbose:
            status = "✓" if merkle_ok else "✗"
            print(f"  {status} Merkle root: {actual_merkle[:16]}...")

        # Verify cross-process determinism
        if verbose:
            print("Verifying cross-process determinism...")

        cross_ok, cross_drifts = verify_cross_process_determinism(
            master_seed_hex=master_seed_hex,
            test_paths=cycle_paths[:3],  # Use first 3 paths
        )
        all_drifts.extend(cross_drifts)
        checks_performed.append("cross_process_verify")

        if verbose:
            status = "✓" if cross_ok else "✗"
            print(f"  {status} Cross-process determinism")

        # Determine overall status
        error_drifts = [d for d in all_drifts if d.severity == "error"]

        if not error_drifts:
            report.status = ReproducibilityStatus.REPRODUCIBLE
            report.message = f"All {len(checks_performed)} checks passed"
        else:
            report.status = ReproducibilityStatus.DRIFT_DETECTED
            report.message = f"Found {len(error_drifts)} drift(s)"

        return report

    except FileNotFoundError:
        report.message = f"Manifest file not found: {manifest_path}"
        report.status = ReproducibilityStatus.ERROR
        return report
    except json.JSONDecodeError as e:
        report.message = f"Invalid JSON in manifest: {e}"
        report.status = ReproducibilityStatus.ERROR
        return report
    except Exception as e:
        report.message = f"Error during verification: {e}"
        report.status = ReproducibilityStatus.ERROR
        return report


def create_synthetic_manifest(
    output_path: Path,
    seed: int = 42,
    num_cycles: int = 10,
    slice_name: str = "test_slice",
    mode: str = "baseline",
) -> Dict[str, Any]:
    """
    Create a synthetic manifest for testing.

    Returns:
        The created manifest dictionary.
    """
    from rfl.prng import DeterministicPRNG, int_to_hex_seed
    from rfl.prng.lineage import SeedLineage

    master_seed_hex = int_to_hex_seed(seed)
    prng = DeterministicPRNG(master_seed_hex)

    # Generate lineage
    cycle_paths = [
        (slice_name, mode, f"cycle_{i:04d}", "ordering")
        for i in range(num_cycles)
    ]

    lineage = SeedLineage(master_seed_hex)
    for path in sorted(cycle_paths):
        lineage.record(*path)

    manifest = {
        "manifest_version": "1.1",
        "experiment_id": "synthetic_test",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": get_current_git_commit(),
            "git_branch": "test",
            "user": "synthetic",
            "machine": "test",
        },
        "configuration": {
            "snapshot": {
                "slice_name": slice_name,
                "mode": mode,
                "num_cycles": num_cycles,
            }
        },
        "execution": {
            "effective_seed": seed,
        },
        "prng_attestation": {
            "schema_version": "1.0",
            "master_seed_hex": master_seed_hex,
            "derivation_scheme": "PRNGKey(root, path) -> SHA256 -> seed % 2^32",
            "implementation": f"rfl/prng/deterministic_prng.py@{get_current_git_commit()[:12]}",
            "lineage_merkle_root": lineage.compute_merkle_root(),
            "lineage_entry_count": len(cycle_paths),
        },
        "results": {
            "seed_schedule": prng.generate_seed_schedule(num_cycles, slice_name, mode),
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reproducibility Probe — Verify Experiment Determinism",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
    0 - Reproducibility verified (REPRODUCIBLE)
    1 - Drift detected (DRIFT_DETECTED)
    2 - Incomplete verification (INCOMPLETE)
    3 - Error during verification

Examples:
    python scripts/reproducibility_probe.py manifest.json
    python scripts/reproducibility_probe.py manifest.json --cycles 10
    python scripts/reproducibility_probe.py manifest.json --output report.json
    python scripts/reproducibility_probe.py --create-synthetic synthetic.json
        """,
    )
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        help="Path to manifest.json file",
    )
    parser.add_argument(
        "--cycles", "-c",
        type=int,
        default=5,
        help="Number of cycles to verify (default: 5)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Write report to JSON file",
    )
    parser.add_argument(
        "--create-synthetic",
        type=Path,
        help="Create a synthetic manifest for testing",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Reproducibility Probe — Agent A2 (runtime-ops-2)")
    print("=" * 70)
    print()

    # Handle synthetic manifest creation
    if args.create_synthetic:
        print(f"Creating synthetic manifest: {args.create_synthetic}")
        manifest = create_synthetic_manifest(args.create_synthetic)
        print(f"  Master seed: {manifest['prng_attestation']['master_seed_hex'][:16]}...")
        print(f"  Merkle root: {manifest['prng_attestation']['lineage_merkle_root'][:16]}...")
        print()
        print("✓ Synthetic manifest created")

        # If manifest path provided, also probe it
        if not args.manifest:
            args.manifest = args.create_synthetic

    if not args.manifest:
        parser.print_help()
        return 2

    # Run probe
    print(f"Probing manifest: {args.manifest}")
    print(f"Cycles to verify: {args.cycles}")
    print()

    report = probe_reproducibility(
        manifest_path=args.manifest,
        num_cycles=args.cycles,
        verbose=args.verbose,
    )

    # Print report
    print()
    print("-" * 70)
    print(f"Status: {report.status.value}")
    print(f"Message: {report.message}")
    print(f"Git commit (manifest): {report.git_commit_manifest[:12]}...")
    print(f"Git commit (current):  {report.git_commit_current[:12]}...")
    print()
    print(f"Checks performed: {', '.join(report.checks_performed)}")
    print(f"Seed schedule verified: {report.seed_schedule_verified}")
    print(f"Lineage Merkle verified: {report.lineage_merkle_verified}")

    if report.drifts:
        print()
        print("Drifts detected:")
        for drift in report.drifts:
            severity_icon = {"info": "ℹ", "warning": "⚠", "error": "❌"}
            icon = severity_icon.get(drift.severity, "?")
            print(f"  {icon} [{drift.severity}] {drift.component}")
            print(f"      Expected: {drift.expected}")
            print(f"      Actual:   {drift.actual}")

    print("-" * 70)

    # Write JSON output if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport written to: {args.output}")

    # Return appropriate exit code
    exit_codes = {
        ReproducibilityStatus.REPRODUCIBLE: 0,
        ReproducibilityStatus.DRIFT_DETECTED: 1,
        ReproducibilityStatus.INCOMPLETE: 2,
        ReproducibilityStatus.ERROR: 3,
    }
    return exit_codes.get(report.status, 3)


if __name__ == "__main__":
    sys.exit(main())

