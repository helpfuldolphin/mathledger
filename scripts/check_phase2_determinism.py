#!/usr/bin/env python3
# PHASE II — NOT USED IN PHASE I
"""
Phase II Determinism Gate — CI Check for PRNG Compliance.

This script performs static and runtime checks to verify that all Phase II
code uses the DeterministicPRNG framework correctly. It is designed to be
run in CI pipelines to catch determinism regressions.

Checks Performed:
    1. Static Analysis: No Phase II module imports/calls global random functions
    2. Manifest Attestation: All manifests have valid prng_attestation blocks
    3. Cross-Process Verification: PRNG produces identical output across processes
    4. Integrity Guard: Runtime detection of banned patterns

Exit Codes:
    0 - All checks passed
    1 - One or more checks failed
    2 - Error during check execution

Usage:
    python scripts/check_phase2_determinism.py
    python scripts/check_phase2_determinism.py --verbose
    python scripts/check_phase2_determinism.py --check-manifests artifacts/

Author: Agent A2 (runtime-ops-2)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CheckResult:
    """Result of a single check."""
    name: str
    passed: bool
    message: str
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class GateResult:
    """Aggregate result of all checks."""
    passed: bool
    checks: List[CheckResult]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks],
        }


def check_static_analysis(verbose: bool = False) -> CheckResult:
    """
    Check 1: Static analysis for banned randomness patterns.

    Uses the PRNG integrity guard's audit function to scan all Phase II
    modules for direct random.* or np.random.* usage.
    """
    try:
        from rfl.prng import audit_phase_ii_modules

        result = audit_phase_ii_modules()

        # Count suppressed violations for reporting
        suppressed_count = sum(
            len(m.get("suppressed", []))
            for m in result.get("details", [])
        )

        if result["overall_compliant"]:
            msg = f"All {result['modules_checked']} Phase II modules are PRNG-compliant"
            if suppressed_count > 0:
                msg += f" ({suppressed_count} suppressed)"
            return CheckResult(
                name="static_analysis",
                passed=True,
                message=msg,
            )
        else:
            details = []
            for module in result.get("details", []):
                for v in module.get("violations", []):
                    details.append(
                        f"{module['path']}:{v['line']}: {v['function']} ({v['type']})"
                    )
            return CheckResult(
                name="static_analysis",
                passed=False,
                message=f"Found {result['violations_found']} PRNG violations in Phase II modules",
                details=details[:20],  # Limit to first 20
            )

    except Exception as e:
        import traceback
        return CheckResult(
            name="static_analysis",
            passed=False,
            message=f"Static analysis failed with error: {e}",
            details=[traceback.format_exc()],
        )


def check_cross_process_determinism(verbose: bool = False) -> CheckResult:
    """
    Check 2: Cross-process PRNG determinism.

    Spawns a subprocess and verifies that PRNG produces identical seeds.
    """
    try:
        from rfl.prng import DeterministicPRNG, int_to_hex_seed

        master_seed = int_to_hex_seed(42)
        test_paths = [
            ("slice_a", "baseline", "cycle_0001"),
            ("slice_b", "rfl", "cycle_0002"),
        ]

        # In-process computation
        prng = DeterministicPRNG(master_seed)
        inprocess_seeds = [prng.seed_for_path(*p) for p in test_paths]

        # Subprocess computation
        script = f'''
import sys
sys.path.insert(0, "{PROJECT_ROOT}")
from rfl.prng import DeterministicPRNG
prng = DeterministicPRNG("{master_seed}")
paths = {test_paths}
seeds = [prng.seed_for_path(*p) for p in paths]
print(",".join(str(s) for s in seeds))
'''
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=PROJECT_ROOT,
        )

        if result.returncode != 0:
            return CheckResult(
                name="cross_process_determinism",
                passed=False,
                message=f"Subprocess failed: {result.stderr}",
            )

        subprocess_seeds = [int(s) for s in result.stdout.strip().split(",")]

        if inprocess_seeds == subprocess_seeds:
            return CheckResult(
                name="cross_process_determinism",
                passed=True,
                message="PRNG produces identical seeds across processes",
            )
        else:
            return CheckResult(
                name="cross_process_determinism",
                passed=False,
                message="PRNG seed mismatch between processes",
                details=[
                    f"In-process: {inprocess_seeds}",
                    f"Subprocess: {subprocess_seeds}",
                ],
            )

    except Exception as e:
        return CheckResult(
            name="cross_process_determinism",
            passed=False,
            message=f"Cross-process check failed: {e}",
        )


def check_manifest_attestation(
    manifest_dir: Optional[Path] = None,
    verbose: bool = False,
) -> CheckResult:
    """
    Check 3: Manifest prng_attestation block verification.

    Scans manifests for valid prng_attestation blocks and verifies
    their internal consistency.
    """
    if manifest_dir is None:
        # Default locations to check
        manifest_dirs = [
            PROJECT_ROOT / "artifacts" / "experiments" / "rfl",
            PROJECT_ROOT / "artifacts" / "rfl",
        ]
    else:
        manifest_dirs = [Path(manifest_dir)]

    manifests_checked = 0
    manifests_valid = 0
    issues: List[str] = []

    for dir_path in manifest_dirs:
        if not dir_path.exists():
            continue

        for manifest_file in dir_path.rglob("manifest*.json"):
            manifests_checked += 1

            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

                # Check for prng_attestation block
                attestation = manifest.get("prng_attestation") or manifest.get("prng")
                if not attestation:
                    issues.append(f"{manifest_file}: Missing prng_attestation block")
                    continue

                # Validate required fields
                required_fields = ["master_seed_hex", "derivation_scheme"]
                missing = [f for f in required_fields if f not in attestation]
                if missing:
                    issues.append(
                        f"{manifest_file}: Missing fields: {', '.join(missing)}"
                    )
                    continue

                # Validate master_seed_hex format
                seed_hex = attestation.get("master_seed_hex", "")
                if len(seed_hex) != 64:
                    issues.append(
                        f"{manifest_file}: Invalid master_seed_hex length: {len(seed_hex)}"
                    )
                    continue

                try:
                    int(seed_hex, 16)
                except ValueError:
                    issues.append(
                        f"{manifest_file}: Invalid master_seed_hex format"
                    )
                    continue

                manifests_valid += 1

            except json.JSONDecodeError as e:
                issues.append(f"{manifest_file}: Invalid JSON: {e}")
            except Exception as e:
                issues.append(f"{manifest_file}: Error: {e}")

    if manifests_checked == 0:
        return CheckResult(
            name="manifest_attestation",
            passed=True,
            message="No manifests found to check (OK for new repos)",
        )

    if manifests_valid == manifests_checked:
        return CheckResult(
            name="manifest_attestation",
            passed=True,
            message=f"All {manifests_checked} manifests have valid prng_attestation",
        )
    else:
        return CheckResult(
            name="manifest_attestation",
            passed=False,
            message=f"{manifests_checked - manifests_valid}/{manifests_checked} manifests have issues",
            details=issues[:10],  # Limit to first 10
        )


def check_lineage_merkle_consistency(verbose: bool = False) -> CheckResult:
    """
    Check 4: Lineage Merkle root consistency.

    Verifies that the SeedLineage Merkle root computation is deterministic.
    """
    try:
        from rfl.prng.lineage import SeedLineage

        master_seed = "a" * 64
        paths = [
            ("slice_a", "baseline", "cycle_0001"),
            ("slice_a", "baseline", "cycle_0002"),
            ("slice_b", "rfl", "cycle_0001"),
        ]

        # Compute Merkle root multiple times
        roots = []
        for _ in range(3):
            lineage = SeedLineage(master_seed)
            for path in sorted(paths):  # Sorted for determinism
                lineage.record(*path)
            roots.append(lineage.compute_merkle_root())

        if len(set(roots)) == 1:
            return CheckResult(
                name="lineage_merkle_consistency",
                passed=True,
                message=f"Merkle root is deterministic: {roots[0][:16]}...",
            )
        else:
            return CheckResult(
                name="lineage_merkle_consistency",
                passed=False,
                message="Merkle root computation is not deterministic",
                details=[f"Root {i}: {r}" for i, r in enumerate(roots)],
            )

    except Exception as e:
        return CheckResult(
            name="lineage_merkle_consistency",
            passed=False,
            message=f"Merkle consistency check failed: {e}",
        )


def check_seed_receipt_verification(verbose: bool = False) -> CheckResult:
    """
    Check 5: Seed receipt creation and verification.

    Ensures receipts can be created and verified correctly.
    """
    try:
        from rfl.prng.lineage import SeedReceipt

        master_seed = "b" * 64
        paths = [
            ("test", "receipt", "001"),
            ("test", "receipt", "002"),
        ]

        for path in paths:
            receipt = SeedReceipt.create(master_seed, path)

            if not receipt.verify():
                return CheckResult(
                    name="seed_receipt_verification",
                    passed=False,
                    message=f"Receipt verification failed for path {path}",
                )

            # Test JSON round-trip
            json_str = receipt.to_json()
            restored = SeedReceipt.from_json(json_str)

            if not restored.verify():
                return CheckResult(
                    name="seed_receipt_verification",
                    passed=False,
                    message=f"Receipt round-trip failed for path {path}",
                )

            if restored.derived_seed != receipt.derived_seed:
                return CheckResult(
                    name="seed_receipt_verification",
                    passed=False,
                    message=f"Seed mismatch after round-trip for path {path}",
                )

        return CheckResult(
            name="seed_receipt_verification",
            passed=True,
            message=f"All {len(paths)} receipts created, verified, and round-tripped",
        )

    except Exception as e:
        return CheckResult(
            name="seed_receipt_verification",
            passed=False,
            message=f"Receipt verification check failed: {e}",
        )


def run_all_checks(
    manifest_dir: Optional[Path] = None,
    verbose: bool = False,
) -> GateResult:
    """
    Run all determinism checks and return aggregate result.
    """
    checks = [
        check_static_analysis(verbose),
        check_cross_process_determinism(verbose),
        check_manifest_attestation(manifest_dir, verbose),
        check_lineage_merkle_consistency(verbose),
        check_seed_receipt_verification(verbose),
    ]

    passed = all(c.passed for c in checks)
    passed_count = sum(1 for c in checks if c.passed)
    total_count = len(checks)

    if passed:
        summary = f"✅ All {total_count} determinism checks passed"
    else:
        failed_names = [c.name for c in checks if not c.passed]
        summary = f"❌ {total_count - passed_count}/{total_count} checks failed: {', '.join(failed_names)}"

    return GateResult(passed=passed, checks=checks, summary=summary)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase II Determinism Gate — CI Check for PRNG Compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
    0 - All checks passed
    1 - One or more checks failed
    2 - Error during check execution

Examples:
    python scripts/check_phase2_determinism.py
    python scripts/check_phase2_determinism.py --verbose
    python scripts/check_phase2_determinism.py --check-manifests artifacts/
    python scripts/check_phase2_determinism.py --output results.json
        """,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--check-manifests",
        type=Path,
        help="Directory containing manifests to check",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Write results to JSON file",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Phase II Determinism Gate — Agent A2 (runtime-ops-2)")
    print("=" * 70)
    print()

    try:
        result = run_all_checks(
            manifest_dir=args.check_manifests,
            verbose=args.verbose,
        )

        # Print results
        for check in result.checks:
            status = "✅ PASS" if check.passed else "❌ FAIL"
            print(f"[{status}] {check.name}")
            print(f"         {check.message}")
            if args.verbose and check.details:
                for detail in check.details:
                    print(f"         - {detail}")
            print()

        print("=" * 70)
        print(result.summary)
        print("=" * 70)

        # Write JSON output if requested
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nResults written to: {args.output}")

        return 0 if result.passed else 1

    except Exception as e:
        print(f"\n❌ Error during check execution: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())

