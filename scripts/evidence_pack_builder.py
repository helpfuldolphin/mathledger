#!/usr/bin/env python3
"""
Evidence Pack Builder CLI

Command-line tool for generating Evidence Pack bundles from First Light runs.

Usage:
    python scripts/evidence_pack_builder.py --run-dir results/first_light/run_123
    python scripts/evidence_pack_builder.py --run-dir results/first_light/run_123 --output-dir evidence_packs/
    python scripts/evidence_pack_builder.py --run-dir results/first_light/run_123 --verify

SHADOW MODE CONTRACT:
- All governance checks are advisory only
- Pack generation succeeds even with warnings
- No enforcement or blocking occurs

See docs/system_law/Evidence_Pack_Spec_PhaseX.md for specification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.topology.first_light.evidence_pack import (
    build_evidence_pack,
    verify_merkle_root,
    EvidencePackResult,
    EVIDENCE_PACK_VERSION,
)


def print_result(result: EvidencePackResult, verbose: bool = False) -> None:
    """Print evidence pack result to console."""
    if result.success:
        print(f"\n{'='*60}")
        print("EVIDENCE PACK GENERATED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Bundle ID:    {result.bundle_id}")
        print(f"Merkle Root:  {result.merkle_root}")
        print(f"Manifest:     {result.manifest_path}")
        print(f"Artifacts:    {len(result.artifacts)}")
    else:
        print(f"\n{'='*60}")
        print("EVIDENCE PACK GENERATION FAILED")
        print(f"{'='*60}")
        print(f"Bundle ID:    {result.bundle_id}")

    # Completeness summary
    print(f"\n--- Completeness ---")
    print(f"All Required Present: {result.completeness.all_required_present}")
    if result.completeness.missing_artifacts:
        print(f"Missing Artifacts:")
        for missing in result.completeness.missing_artifacts:
            print(f"  - {missing}")

    # Governance advisories (SHADOW MODE)
    if result.governance_advisories:
        print(f"\n--- Governance Advisories (SHADOW MODE - Advisory Only) ---")
        for advisory in result.governance_advisories:
            status = "PASS" if advisory.passed else "ADVISORY"
            print(f"  [{advisory.severity}] {advisory.check_name}: {status}")
            if not advisory.passed:
                print(f"           {advisory.message}")

    # Errors
    if result.errors:
        print(f"\n--- Errors ---")
        for error in result.errors:
            print(f"  ERROR: {error}")

    # Warnings
    if result.warnings:
        print(f"\n--- Warnings ---")
        for warning in result.warnings:
            print(f"  {warning}")

    # Verbose artifact list
    if verbose and result.artifacts:
        print(f"\n--- Artifacts ({len(result.artifacts)}) ---")
        for artifact in result.artifacts:
            status = "OK" if artifact.validation_passed else "INVALID"
            req = "REQ" if artifact.required else "OPT"
            print(f"  [{req}] [{status}] {artifact.path}")
            print(f"          SHA256: {artifact.sha256[:16]}...")
            print(f"          Size: {artifact.size_bytes:,} bytes")
            if artifact.validation_errors:
                for error in artifact.validation_errors:
                    print(f"          Error: {error}")

    print()


def cmd_build(args: argparse.Namespace) -> int:
    """Build an evidence pack."""
    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        print(f"Error: Run directory does not exist: {run_dir}")
        return 1

    print(f"Building Evidence Pack v{EVIDENCE_PACK_VERSION}")
    print(f"Run Directory: {run_dir}")

    if args.output_dir:
        print(f"Output Directory: {args.output_dir}")

    result = build_evidence_pack(
        run_dir=run_dir,
        output_dir=args.output_dir,
        schemas_dir=args.schemas_dir,
        validate_schemas=not args.skip_validation,
        p3_run_id=args.p3_run_id,
        p4_run_id=args.p4_run_id,
    )

    print_result(result, verbose=args.verbose)

    # Return code based on success
    if not result.success:
        return 1

    # Return warning code if governance checks failed
    failed_advisories = [a for a in result.governance_advisories if not a.passed]
    if failed_advisories and args.strict:
        print("Strict mode: Governance advisories failed")
        return 2

    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a manifest's Merkle root."""
    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        print(f"Error: Manifest does not exist: {manifest_path}")
        return 1

    print(f"Verifying Merkle root: {manifest_path}")

    is_valid, message = verify_merkle_root(manifest_path)

    if is_valid:
        print(f"VERIFIED: {message}")
        return 0
    else:
        print(f"FAILED: {message}")
        return 1


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect a manifest file."""
    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        print(f"Error: Manifest does not exist: {manifest_path}")
        return 1

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error: Failed to load manifest: {e}")
        return 1

    print(f"\n{'='*60}")
    print("EVIDENCE PACK MANIFEST")
    print(f"{'='*60}")
    print(f"Bundle ID:      {manifest.get('bundle_id', 'N/A')}")
    print(f"Bundle Version: {manifest.get('bundle_version', 'N/A')}")
    print(f"Generated At:   {manifest.get('generated_at', 'N/A')}")
    print(f"P3 Run ID:      {manifest.get('p3_run_id', 'N/A')}")
    print(f"P4 Run ID:      {manifest.get('p4_run_id', 'N/A')}")
    print(f"Merkle Root:    {manifest.get('cryptographic_root', 'N/A')}")

    artifacts = manifest.get("artifacts", [])
    print(f"\nArtifacts: {len(artifacts)}")

    if args.verbose:
        for artifact in artifacts:
            req = "REQ" if artifact.get("required", False) else "OPT"
            print(f"  [{req}] {artifact.get('path', 'N/A')}")
            print(f"        SHA256: {artifact.get('sha256', 'N/A')[:16]}...")

    completeness = manifest.get("completeness", {})
    print(f"\nCompleteness:")
    print(f"  All Required Present: {completeness.get('all_required_present', 'N/A')}")

    missing = completeness.get("missing_artifacts", [])
    if missing:
        print(f"  Missing: {', '.join(missing)}")

    validation = manifest.get("validation_status", {})
    print(f"\nValidation Status:")
    print(f"  All Artifacts Present:  {validation.get('all_artifacts_present', 'N/A')}")
    print(f"  All Hashes Verified:    {validation.get('all_hashes_verified', 'N/A')}")
    print(f"  All Schemas Valid:      {validation.get('all_schemas_valid', 'N/A')}")
    print(f"  Completeness Passed:    {validation.get('completeness_check_passed', 'N/A')}")

    print()
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evidence Pack Builder CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build evidence pack from run directory
  python scripts/evidence_pack_builder.py build --run-dir results/first_light/run_123

  # Build with custom output directory
  python scripts/evidence_pack_builder.py build --run-dir results/first_light/run_123 --output-dir evidence_packs/

  # Verify existing manifest
  python scripts/evidence_pack_builder.py verify --manifest evidence_packs/manifest.json

  # Inspect manifest contents
  python scripts/evidence_pack_builder.py inspect --manifest evidence_packs/manifest.json --verbose

SHADOW MODE CONTRACT:
  All governance checks are advisory only. Pack generation succeeds even with warnings.
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"Evidence Pack Builder v{EVIDENCE_PACK_VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build an evidence pack")
    build_parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to First Light run directory",
    )
    build_parser.add_argument(
        "--output-dir",
        help="Output directory for manifest (defaults to run-dir)",
    )
    build_parser.add_argument(
        "--schemas-dir",
        help="Path to schema files (defaults to docs/system_law/schemas)",
    )
    build_parser.add_argument(
        "--p3-run-id",
        help="P3 run identifier for manifest",
    )
    build_parser.add_argument(
        "--p4-run-id",
        help="P4 run identifier for manifest",
    )
    build_parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip JSON schema validation",
    )
    build_parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code if governance checks fail",
    )
    build_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a manifest's Merkle root")
    verify_parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest.json file",
    )

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a manifest file")
    inspect_parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest.json file",
    )
    inspect_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "build":
        return cmd_build(args)
    elif args.command == "verify":
        return cmd_verify(args)
    elif args.command == "inspect":
        return cmd_inspect(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
