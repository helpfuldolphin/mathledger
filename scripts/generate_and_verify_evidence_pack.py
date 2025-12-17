#!/usr/bin/env python3
"""
Unified Evidence Pack Generator and Verifier

ONE COMMAND for external evaluators to generate, verify, and receive
a compliance verdict for CAL-EXP-3 / First-Light evidence packs.

SHADOW MODE CONTRACT:
- This script orchestrates existing tools (no new logic)
- All artifacts retain mode="SHADOW" markers
- Failures are reported loudly, never hidden

Usage:
    # Generate + verify (default: uses existing artifacts if present)
    python scripts/generate_and_verify_evidence_pack.py

    # Verify only (requires existing evidence pack)
    python scripts/generate_and_verify_evidence_pack.py --verify-only

    # Force regeneration of all artifacts
    python scripts/generate_and_verify_evidence_pack.py --regenerate

    # Custom output directory
    python scripts/generate_and_verify_evidence_pack.py --output-dir results/my_pack

Exit Codes:
    0 = Evidence pack generated and verified successfully
    1 = Verification failed (missing/mismatched files)
    2 = Generation failed (missing source artifacts)
    3 = Configuration error
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================================
# Constants
# ============================================================================

DEFAULT_P3_SEARCH_PATHS = [
    Path("results/first_light/golden_run/p3"),
    Path("results/verification/p3"),
    Path("results/p3"),
]

DEFAULT_P4_SEARCH_PATHS = [
    Path("results/first_light/golden_run/p4"),
    Path("results/verification/p4"),
    Path("results/p4"),
]

DEFAULT_OUTPUT_DIR = Path("results/first_light/evidence_pack_first_light")

CAL_EXP_SEARCH_PATHS = [
    Path("results/calibration"),
    Path("results/first_light/calibration"),
    Path("artifacts/calibration"),
]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DiscoveryResult:
    """Result of artifact discovery phase."""
    p3_dir: Optional[Path] = None
    p4_dir: Optional[Path] = None
    cal_exp1_report: Optional[Path] = None
    cal_exp2_report: Optional[Path] = None
    cal_exp3_report: Optional[Path] = None
    ledger_guard_summary: Optional[Path] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_minimum_artifacts(self) -> bool:
        """Check if minimum required artifacts are present."""
        return self.p3_dir is not None and self.p4_dir is not None


@dataclass
class ComplianceReport:
    """Final compliance report for evaluators."""
    schema_version: str = "1.0.0"
    mode: str = "SHADOW"
    generated_at: str = ""

    # Discovery
    discovery_status: str = "UNKNOWN"
    p3_dir: str = ""
    p4_dir: str = ""
    artifacts_discovered: Dict[str, bool] = field(default_factory=dict)

    # Generation
    generation_status: str = "UNKNOWN"
    evidence_pack_dir: str = ""
    manifest_path: str = ""

    # Verification
    verification_status: str = "UNKNOWN"
    total_files: int = 0
    verified_files: int = 0
    missing_files: List[str] = field(default_factory=list)
    mismatched_files: List[str] = field(default_factory=list)
    manifest_sha256: str = ""

    # Final verdict
    verdict: str = "UNKNOWN"  # "PASS", "FAIL", "ERROR"
    verdict_reason: str = ""

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "generated_at": self.generated_at,
            "discovery": {
                "status": self.discovery_status,
                "p3_dir": self.p3_dir,
                "p4_dir": self.p4_dir,
                "artifacts_discovered": self.artifacts_discovered,
            },
            "generation": {
                "status": self.generation_status,
                "evidence_pack_dir": self.evidence_pack_dir,
                "manifest_path": self.manifest_path,
            },
            "verification": {
                "status": self.verification_status,
                "total_files": self.total_files,
                "verified_files": self.verified_files,
                "missing_files": self.missing_files,
                "mismatched_files": self.mismatched_files,
                "manifest_sha256": self.manifest_sha256,
            },
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ============================================================================
# Discovery Functions
# ============================================================================

def find_artifacts_parent_dir(search_paths: List[Path], artifact_prefix: str) -> Optional[Path]:
    """
    Find the parent directory containing run artifacts.

    The build_first_light_evidence_pack.py script expects:
    - --p3-dir = parent directory containing fl_* subdirectories
    - --p4-dir = parent directory containing p4_* subdirectories

    This function returns the PARENT directory, not the run directory itself.

    Args:
        search_paths: List of paths to search
        artifact_prefix: Prefix for run directories (e.g., "fl_" for P3, "p4_" for P4)

    Returns:
        Parent directory containing artifact subdirectories, or None if not found
    """
    for search_path in search_paths:
        if not search_path.exists():
            continue

        # Check if this directory contains artifact subdirectories
        subdirs = list(search_path.glob(f"{artifact_prefix}*"))
        if subdirs:
            # Found a parent directory with artifact subdirectories
            return search_path

    return None


def find_latest_run_dir(search_paths: List[Path]) -> Optional[Path]:
    """
    Find the latest run directory from search paths.

    Looks for directories containing required artifacts, preferring
    the most recently modified.
    """
    candidates: List[tuple[Path, float]] = []

    for search_path in search_paths:
        if not search_path.exists():
            continue

        # Check if this is a run directory directly
        if (search_path / "stability_report.json").exists() or \
           (search_path / "p4_summary.json").exists():
            mtime = search_path.stat().st_mtime
            candidates.append((search_path, mtime))
            continue

        # Check subdirectories (e.g., fl_20251211_044905_seed42)
        for subdir in search_path.iterdir():
            if not subdir.is_dir():
                continue
            if (subdir / "stability_report.json").exists() or \
               (subdir / "p4_summary.json").exists() or \
               (subdir / "synthetic_raw.jsonl").exists() or \
               (subdir / "real_cycles.jsonl").exists():
                mtime = subdir.stat().st_mtime
                candidates.append((subdir, mtime))

    if not candidates:
        return None

    # Return most recent
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def find_cal_exp_report(report_name: str) -> Optional[Path]:
    """Find a CAL-EXP report file."""
    for search_path in CAL_EXP_SEARCH_PATHS:
        candidate = search_path / report_name
        if candidate.exists():
            return candidate
    return None


def discover_artifacts(
    p3_dir: Optional[Path] = None,
    p4_dir: Optional[Path] = None,
) -> DiscoveryResult:
    """
    Discover existing CAL-EXP-3 / First-Light artifacts.

    NOTE: The build_first_light_evidence_pack.py script expects PARENT directories
    that contain fl_*/p4_* subdirectories, not the run directories themselves.

    Args:
        p3_dir: Explicit P3 parent directory (skips auto-discovery if provided)
        p4_dir: Explicit P4 parent directory (skips auto-discovery if provided)

    Returns:
        DiscoveryResult with found paths and any errors
    """
    result = DiscoveryResult()

    # Discover P3 artifacts (parent directory containing fl_* subdirs)
    if p3_dir is not None:
        if p3_dir.exists():
            result.p3_dir = p3_dir
        else:
            result.errors.append(f"Specified P3 directory does not exist: {p3_dir}")
    else:
        result.p3_dir = find_artifacts_parent_dir(DEFAULT_P3_SEARCH_PATHS, "fl_")
        if result.p3_dir is None:
            result.errors.append(
                "No P3 artifacts found. Searched: " +
                ", ".join(str(p) for p in DEFAULT_P3_SEARCH_PATHS)
            )

    # Discover P4 artifacts (parent directory containing p4_* subdirs)
    if p4_dir is not None:
        if p4_dir.exists():
            result.p4_dir = p4_dir
        else:
            result.errors.append(f"Specified P4 directory does not exist: {p4_dir}")
    else:
        result.p4_dir = find_artifacts_parent_dir(DEFAULT_P4_SEARCH_PATHS, "p4_")
        if result.p4_dir is None:
            result.errors.append(
                "No P4 artifacts found. Searched: " +
                ", ".join(str(p) for p in DEFAULT_P4_SEARCH_PATHS)
            )

    # Discover CAL-EXP reports (optional)
    result.cal_exp1_report = find_cal_exp_report("cal_exp1_report.json")
    result.cal_exp2_report = find_cal_exp_report("cal_exp2_report.json")
    result.cal_exp3_report = find_cal_exp_report("cal_exp3_report.json")

    # Discover ledger guard summary (optional)
    for search_path in CAL_EXP_SEARCH_PATHS:
        candidate = search_path / "ledger_guard_summary.json"
        if candidate.exists():
            result.ledger_guard_summary = candidate
            break

    # Add warnings for missing optional artifacts
    if result.cal_exp3_report is None:
        result.warnings.append("CAL-EXP-3 report not found (optional)")

    return result


# ============================================================================
# Generation Functions
# ============================================================================

def generate_evidence_pack(
    discovery: DiscoveryResult,
    output_dir: Path,
) -> tuple[bool, str]:
    """
    Generate evidence pack using existing build script.

    Args:
        discovery: Discovery result with artifact paths
        output_dir: Output directory for evidence pack

    Returns:
        Tuple of (success, error_message)
    """
    if not discovery.has_minimum_artifacts:
        return False, "Missing minimum required artifacts (P3 and P4 directories)"

    # Build command
    cmd = [
        sys.executable,
        "scripts/build_first_light_evidence_pack.py",
        "--p3-dir", str(discovery.p3_dir),
        "--p4-dir", str(discovery.p4_dir),
        "--output-dir", str(output_dir),
    ]

    # Add optional CAL-EXP reports
    if discovery.cal_exp1_report:
        cmd.extend(["--cal-exp1-report", str(discovery.cal_exp1_report)])
    if discovery.cal_exp2_report:
        cmd.extend(["--cal-exp2-report", str(discovery.cal_exp2_report)])
    if discovery.cal_exp3_report:
        cmd.extend(["--cal-exp3-report", str(discovery.cal_exp3_report)])

    # Add ledger guard summary
    if discovery.ledger_guard_summary:
        cmd.extend(["--ledger-guard-summary", str(discovery.ledger_guard_summary)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            return False, f"Build script failed: {error_msg}"

        # Verify manifest was created
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            return False, "Build completed but manifest.json not found"

        return True, ""

    except subprocess.TimeoutExpired:
        return False, "Build script timed out (>5 minutes)"
    except Exception as e:
        return False, f"Build script error: {e}"


# ============================================================================
# Verification Functions
# ============================================================================

def verify_evidence_pack(pack_dir: Path) -> Dict[str, Any]:
    """
    Verify evidence pack integrity using existing verify script.

    Args:
        pack_dir: Path to evidence pack directory

    Returns:
        Verification report dictionary
    """
    # Import verification function directly for cleaner integration
    try:
        from scripts.verify_evidence_pack_integrity import verify_evidence_pack as _verify
        report = _verify(pack_dir)
        return report.to_dict()
    except ImportError:
        # Fallback to subprocess
        cmd = [
            sys.executable,
            "scripts/verify_evidence_pack_integrity.py",
            "--pack-dir", str(pack_dir),
            "--json-output", str(pack_dir / "_integrity_report.json"),
            "--quiet",
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            report_path = pack_dir / "_integrity_report.json"
            if report_path.exists():
                with open(report_path) as f:
                    return json.load(f)
        except Exception:
            pass

        return {"status": "ERROR", "errors": ["Verification script failed"]}


# ============================================================================
# Main Orchestration
# ============================================================================

def sign_manifest(manifest_path: Path, signing_key_path: Path) -> tuple[bool, str]:
    """
    Sign the manifest using Ed25519.

    Args:
        manifest_path: Path to manifest.json
        signing_key_path: Path to Ed25519 private key (PEM)

    Returns:
        Tuple of (success, error_or_signature_path)
    """
    try:
        from scripts.sign_manifest import load_private_key, sign_file

        private_key = load_private_key(signing_key_path)
        signature = sign_file(manifest_path, private_key)

        signature_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")
        with open(signature_path, "wb") as f:
            f.write(signature)

        return True, str(signature_path)
    except Exception as e:
        return False, str(e)


def verify_manifest_signature(
    manifest_path: Path,
    pubkey_path: Path,
) -> tuple[bool, str]:
    """
    Verify manifest signature.

    Args:
        manifest_path: Path to manifest.json
        pubkey_path: Path to Ed25519 public key (PEM)

    Returns:
        Tuple of (is_valid, status_message)
    """
    signature_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")

    if not signature_path.exists():
        return False, "Signature file not found"

    try:
        from scripts.verify_manifest_signature import load_public_key, verify_signature

        public_key = load_public_key(pubkey_path)
        is_valid = verify_signature(manifest_path, signature_path, public_key)

        if is_valid:
            return True, "VERIFIED"
        else:
            return False, "INVALID"
    except Exception as e:
        return False, f"Verification error: {e}"


def run_pipeline(
    output_dir: Path,
    p3_dir: Optional[Path] = None,
    p4_dir: Optional[Path] = None,
    verify_only: bool = False,
    regenerate: bool = False,
    sign: bool = False,
    signing_key: Optional[Path] = None,
    verify_sig: bool = False,
    pubkey: Optional[Path] = None,
) -> ComplianceReport:
    """
    Run the full evidence pack pipeline.

    Args:
        output_dir: Output directory for evidence pack
        p3_dir: Explicit P3 directory (optional)
        p4_dir: Explicit P4 directory (optional)
        verify_only: Skip generation, only verify existing pack
        regenerate: Force regeneration even if pack exists
        sign: Sign the manifest after generation
        signing_key: Path to Ed25519 private key for signing
        verify_sig: Verify manifest signature
        pubkey: Path to Ed25519 public key for verification

    Returns:
        ComplianceReport with final verdict
    """
    report = ComplianceReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        evidence_pack_dir=str(output_dir),
        manifest_path=str(output_dir / "manifest.json"),
    )

    # ========================================================================
    # Phase 1: Discovery
    # ========================================================================
    print("=" * 60)
    print("PHASE 1: ARTIFACT DISCOVERY")
    print("  (P3 = synthetic stability data, P4 = shadow twin data)")
    print("=" * 60)
    print()

    if verify_only:
        # Skip discovery, just verify existing pack
        if not output_dir.exists():
            report.discovery_status = "SKIPPED"
            report.errors.append(f"Evidence pack directory does not exist: {output_dir}")
            report.verdict = "ERROR"
            report.verdict_reason = "Cannot verify non-existent evidence pack"
            return report

        report.discovery_status = "SKIPPED"
        print(f"  Verify-only mode: using existing pack at {output_dir}")
        print()
    else:
        discovery = discover_artifacts(p3_dir, p4_dir)

        report.p3_dir = str(discovery.p3_dir) if discovery.p3_dir else ""
        report.p4_dir = str(discovery.p4_dir) if discovery.p4_dir else ""
        report.artifacts_discovered = {
            "p3_dir": discovery.p3_dir is not None,
            "p4_dir": discovery.p4_dir is not None,
            "cal_exp1_report": discovery.cal_exp1_report is not None,
            "cal_exp2_report": discovery.cal_exp2_report is not None,
            "cal_exp3_report": discovery.cal_exp3_report is not None,
            "ledger_guard_summary": discovery.ledger_guard_summary is not None,
        }
        report.warnings.extend(discovery.warnings)

        if discovery.p3_dir:
            print(f"  P3 artifacts: {discovery.p3_dir}")
        else:
            print("  P3 artifacts: NOT FOUND")

        if discovery.p4_dir:
            print(f"  P4 artifacts: {discovery.p4_dir}")
        else:
            print("  P4 artifacts: NOT FOUND")

        if discovery.cal_exp3_report:
            print(f"  CAL-EXP-3 report: {discovery.cal_exp3_report}")
        else:
            print("  CAL-EXP-3 report: not found (optional)")

        print()

        if not discovery.has_minimum_artifacts:
            report.discovery_status = "FAILED"
            report.errors.extend(discovery.errors)
            report.verdict = "ERROR"
            report.verdict_reason = "Missing required P3/P4 artifacts"
            print("DISCOVERY FAILED: Missing required artifacts")
            print()
            for error in discovery.errors:
                print(f"  ERROR: {error}")
            print()
            print("To generate artifacts, run the P3/P4 harnesses:")
            print("  uv run python scripts/usla_first_light_harness.py --cycles 1000 --seed 42 --output-dir results/first_light/golden_run/p3")
            print("  uv run python scripts/usla_first_light_p4_harness.py --cycles 1000 --seed 42 --output-dir results/first_light/golden_run/p4")
            return report

        report.discovery_status = "PASSED"
        print("DISCOVERY: PASSED")
        print()

    # ========================================================================
    # Phase 2: Generation
    # ========================================================================
    print("=" * 60)
    print("PHASE 2: EVIDENCE PACK GENERATION")
    print("=" * 60)
    print()

    manifest_exists = (output_dir / "manifest.json").exists()

    if verify_only:
        report.generation_status = "SKIPPED"
        print("  Skipped (verify-only mode)")
        print()
    elif manifest_exists and not regenerate:
        report.generation_status = "SKIPPED"
        print(f"  Evidence pack already exists at {output_dir}")
        print("  Use --regenerate to force rebuild")
        print()
    else:
        if regenerate and manifest_exists:
            print("  Regenerating evidence pack (--regenerate flag set)")
        else:
            print("  Generating new evidence pack...")
        print()

        success, error = generate_evidence_pack(discovery, output_dir)

        if success:
            report.generation_status = "PASSED"
            print("GENERATION: PASSED")
        else:
            report.generation_status = "FAILED"
            report.errors.append(error)
            report.verdict = "ERROR"
            report.verdict_reason = f"Evidence pack generation failed: {error}"
            print(f"GENERATION: FAILED")
            print(f"  ERROR: {error}")
            return report

        print()

    # ========================================================================
    # Phase 3: Verification
    # ========================================================================
    print("=" * 60)
    print("PHASE 3: INTEGRITY VERIFICATION")
    print("=" * 60)
    print()

    verification = verify_evidence_pack(output_dir)

    report.verification_status = verification.get("status", "UNKNOWN")
    report.total_files = verification.get("summary", {}).get("total_files", 0)
    report.verified_files = verification.get("summary", {}).get("ok_count", 0)
    report.missing_files = verification.get("missing_files", [])
    report.mismatched_files = verification.get("mismatch_files", [])
    report.manifest_sha256 = verification.get("manifest_hash", "")

    if verification.get("errors"):
        report.errors.extend(verification["errors"])

    print(f"  Total files in manifest: {report.total_files}")
    print(f"  Verified (OK): {report.verified_files}")
    print(f"  Missing: {len(report.missing_files)}")
    print(f"  Mismatched: {len(report.mismatched_files)}")
    print()

    if report.missing_files:
        print("  Missing files:")
        for f in report.missing_files[:5]:
            print(f"    - {f}")
        if len(report.missing_files) > 5:
            print(f"    ... and {len(report.missing_files) - 5} more")
        print()

    if report.mismatched_files:
        print("  Mismatched files:")
        for f in report.mismatched_files[:5]:
            print(f"    - {f}")
        if len(report.mismatched_files) > 5:
            print(f"    ... and {len(report.mismatched_files) - 5} more")
        print()

    print(f"  Manifest SHA-256: {report.manifest_sha256}")
    print()
    print(f"VERIFICATION: {report.verification_status}")
    print()

    # ========================================================================
    # Phase 4: Manifest Signing (Optional)
    # ========================================================================
    signing_status = "SKIPPED"
    signature_path_str = ""

    if sign and signing_key is not None:
        print("=" * 60)
        print("PHASE 4: MANIFEST SIGNING")
        print("=" * 60)
        print()

        manifest_path = output_dir / "manifest.json"
        success, result = sign_manifest(manifest_path, signing_key)

        if success:
            signing_status = "SIGNED"
            signature_path_str = result
            print(f"  Manifest signed successfully")
            print(f"  Signature: {result}")
        else:
            signing_status = "FAILED"
            report.warnings.append(f"Signing failed: {result}")
            print(f"  Signing failed: {result}")

        print()
        print(f"SIGNING: {signing_status}")
        print()

    # ========================================================================
    # Phase 5: Signature Verification (Optional)
    # ========================================================================
    sig_verification_status = "SKIPPED"

    if verify_sig and pubkey is not None:
        print("=" * 60)
        print("PHASE 5: SIGNATURE VERIFICATION")
        print("=" * 60)
        print()

        manifest_path = output_dir / "manifest.json"
        is_valid, status_msg = verify_manifest_signature(manifest_path, pubkey)

        if is_valid:
            sig_verification_status = "VERIFIED"
            print(f"  Signature verified successfully")
        else:
            sig_verification_status = f"FAILED ({status_msg})"
            report.warnings.append(f"Signature verification failed: {status_msg}")
            print(f"  Signature verification failed: {status_msg}")

        print()
        print(f"SIGNATURE VERIFICATION: {sig_verification_status}")
        print()

    # ========================================================================
    # Final Verdict
    # ========================================================================
    print("=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    print()

    if report.verification_status == "PASSED":
        report.verdict = "PASS"
        report.verdict_reason = (
            f"Evidence pack verified: {report.verified_files}/{report.total_files} files OK"
        )
    elif report.verification_status == "FAILED":
        report.verdict = "FAIL"
        issues = []
        if report.missing_files:
            issues.append(f"{len(report.missing_files)} missing")
        if report.mismatched_files:
            issues.append(f"{len(report.mismatched_files)} mismatched")
        report.verdict_reason = f"Integrity check failed: {', '.join(issues)}"
    else:
        report.verdict = "ERROR"
        report.verdict_reason = "Verification could not complete"

    if report.verdict == "PASS":
        print(f"  VERDICT: {report.verdict}")
        print(f"  {report.verdict_reason}")
        print()
        print("  The evidence pack is ready for external audit.")
        print(f"  Location: {output_dir}")
        print(f"  Manifest: {output_dir / 'manifest.json'}")
    else:
        print(f"  VERDICT: {report.verdict}")
        print(f"  {report.verdict_reason}")
        print()
        if report.errors:
            print("  Errors:")
            for error in report.errors:
                print(f"    - {error}")

    print()
    print("-" * 60)
    print("THIS VERIFIED: File integrity (SHA-256 hashes match manifest)")
    print("NOT VERIFIED:  Lean proofs, determinism, or scientific claims")
    print("-" * 60)
    print("SHADOW MODE: All artifacts are observation-only (calibration phase).")
    print("=" * 60)

    return report


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate and verify CAL-EXP-3 / First-Light evidence pack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate + verify (auto-discovers artifacts)
  python scripts/generate_and_verify_evidence_pack.py

  # Verify existing pack only
  python scripts/generate_and_verify_evidence_pack.py --verify-only

  # Force regeneration
  python scripts/generate_and_verify_evidence_pack.py --regenerate

  # Explicit artifact paths
  python scripts/generate_and_verify_evidence_pack.py \\
      --p3-dir results/my_p3 \\
      --p4-dir results/my_p4

Exit codes:
  0 = PASS (verified successfully)
  1 = FAIL (verification failed)
  2 = ERROR (generation failed)
  3 = Configuration error
""",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for evidence pack (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--p3-dir",
        type=str,
        help="Explicit P3 artifacts directory (skips auto-discovery)",
    )
    parser.add_argument(
        "--p4-dir",
        type=str,
        help="Explicit P4 artifacts directory (skips auto-discovery)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip generation, only verify existing evidence pack",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration even if evidence pack exists",
    )
    parser.add_argument(
        "--json-report",
        type=str,
        help="Path to write JSON compliance report",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable output (use with --json-report)",
    )
    parser.add_argument(
        "--sign",
        action="store_true",
        help="Sign the manifest after generation (requires --signing-key)",
    )
    parser.add_argument(
        "--signing-key",
        type=str,
        help="Path to Ed25519 private key for signing (PEM format)",
    )
    parser.add_argument(
        "--verify-signature",
        action="store_true",
        help="Verify manifest signature (requires --pubkey)",
    )
    parser.add_argument(
        "--pubkey",
        type=str,
        help="Path to Ed25519 public key for signature verification (PEM format)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.verify_only and args.regenerate:
        print("ERROR: --verify-only and --regenerate are mutually exclusive")
        return 3

    if args.sign and not args.signing_key:
        print("ERROR: --sign requires --signing-key")
        return 3

    if args.verify_signature and not args.pubkey:
        print("ERROR: --verify-signature requires --pubkey")
        return 3

    output_dir = Path(args.output_dir)
    p3_dir = Path(args.p3_dir) if args.p3_dir else None
    p4_dir = Path(args.p4_dir) if args.p4_dir else None

    # Suppress stdout if quiet mode
    if args.quiet:
        import io
        sys.stdout = io.StringIO()

    try:
        report = run_pipeline(
            output_dir=output_dir,
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            verify_only=args.verify_only,
            regenerate=args.regenerate,
        )
    finally:
        if args.quiet:
            sys.stdout = sys.__stdout__

    # Write JSON report if requested
    if args.json_report:
        report_path = Path(args.json_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        if not args.quiet:
            print()
            print(f"JSON report written to: {report_path}")

    # Return exit code based on verdict
    if report.verdict == "PASS":
        return 0
    elif report.verdict == "FAIL":
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
