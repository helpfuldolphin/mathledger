#!/usr/bin/env python3
"""
PHASE II — NOT USED IN PHASE I

U2 Uplift Experiment Auditor
============================

One-shot audit script for U2 experiment directories. This script:
- Discovers manifest, baseline log, RFL log, and ht_series.json
- Validates all bindings via ManifestVerifier
- Validates Hₜ series integrity if present
- Validates PHASE II labelling and required fields
- Emits a combined Markdown + JSON report

This auditor does NOT:
- Perform uplift significance testing
- Interpret statistical outcomes
- Modify any attestation files
- Depend on database or network

Usage:
    uv run python experiments/audit_uplift_u2.py results/uplift_u2/slice_uplift_goal

Output:
    - audit_report.md in the experiment directory
    - audit_report.json in the experiment directory

Exit codes:
    0: All checks passed
    1: One or more checks failed
    2: Critical error (e.g., directory not found)
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.manifest_verifier import (
    CheckResult,
    ManifestVerifier,
    VerificationReport,
)

# Phase II label that must appear in all artifacts
PHASE_II_LABEL = "PHASE II — NOT USED IN PHASE I"


def discover_artifacts(
    experiment_dir: Path,
) -> Tuple[
    Optional[Path],  # manifest
    Optional[Path],  # baseline log
    Optional[Path],  # rfl log
    Optional[Path],  # ht_series
    List[str],  # discovery notes
]:
    """
    Auto-discover experiment artifacts in the given directory.

    Returns:
        Tuple of (manifest_path, baseline_log, rfl_log, ht_series, notes)
    """
    notes: List[str] = []
    manifest_path: Optional[Path] = None
    baseline_log: Optional[Path] = None
    rfl_log: Optional[Path] = None
    ht_series: Optional[Path] = None

    if not experiment_dir.exists():
        notes.append(f"ERROR: Directory does not exist: {experiment_dir}")
        return None, None, None, None, notes

    # Discover manifest file
    manifest_candidates = list(experiment_dir.glob("*manifest*.json"))
    if manifest_candidates:
        manifest_path = manifest_candidates[0]
        notes.append(f"Found manifest: {manifest_path.name}")
        if len(manifest_candidates) > 1:
            notes.append(
                f"  (multiple manifests found, using first: {[p.name for p in manifest_candidates]})"
            )
    else:
        notes.append("WARNING: No manifest file found")

    # Discover baseline log
    baseline_candidates = list(experiment_dir.glob("*baseline*.jsonl"))
    if baseline_candidates:
        baseline_log = baseline_candidates[0]
        notes.append(f"Found baseline log: {baseline_log.name}")
    else:
        notes.append("WARNING: No baseline log found")

    # Discover RFL log
    rfl_candidates = list(experiment_dir.glob("*rfl*.jsonl"))
    if rfl_candidates:
        rfl_log = rfl_candidates[0]
        notes.append(f"Found RFL log: {rfl_log.name}")
    else:
        notes.append("WARNING: No RFL log found")

    # Discover ht_series.json
    ht_candidates = list(experiment_dir.glob("*ht_series*.json"))
    if ht_candidates:
        ht_series = ht_candidates[0]
        notes.append(f"Found ht_series: {ht_series.name}")
    else:
        notes.append("INFO: No ht_series.json found (optional)")

    return manifest_path, baseline_log, rfl_log, ht_series, notes


def count_log_records(log_path: Path) -> Dict[str, int]:
    """
    Count records and aggregate statistics from a JSONL log file.

    Returns dict with:
        - total_records
        - cycles (unique cycle numbers)
        - successes (records where success=True)
        - abstentions (records where success=False or abstained)
    """
    counts = {
        "total_records": 0,
        "cycles": 0,
        "successes": 0,
        "abstentions": 0,
    }

    if not log_path or not log_path.exists():
        return counts

    seen_cycles = set()
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                counts["total_records"] += 1
                try:
                    record = json.loads(line)
                    cycle = record.get("cycle")
                    if cycle is not None:
                        seen_cycles.add(cycle)
                    # Count successes/abstentions
                    success = record.get("success")
                    if success is True:
                        counts["successes"] += 1
                    elif success is False:
                        counts["abstentions"] += 1
                except json.JSONDecodeError:
                    pass  # Count record but skip parsing
        counts["cycles"] = len(seen_cycles)
    except Exception:
        pass

    return counts


def check_log_phase_ii_labels(log_path: Path) -> CheckResult:
    """
    Verify that log records contain PHASE II label.

    Returns CheckResult.
    """
    if not log_path or not log_path.exists():
        return CheckResult(
            name=f"log_phase_ii_labels_{log_path.name if log_path else 'unknown'}",
            passed=False,
            message="Log file not found",
        )

    labeled_count = 0
    total_count = 0

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_count += 1
                try:
                    record = json.loads(line)
                    label = record.get("label", "")
                    if PHASE_II_LABEL in label:
                        labeled_count += 1
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        return CheckResult(
            name=f"log_phase_ii_labels_{log_path.name}",
            passed=False,
            message=f"Error reading log: {e}",
        )

    if total_count == 0:
        return CheckResult(
            name=f"log_phase_ii_labels_{log_path.name}",
            passed=False,
            message="Log file is empty",
        )

    if labeled_count == total_count:
        return CheckResult(
            name=f"log_phase_ii_labels_{log_path.name}",
            passed=True,
            message=f"All {total_count} records have PHASE II label",
        )
    elif labeled_count == 0:
        return CheckResult(
            name=f"log_phase_ii_labels_{log_path.name}",
            passed=False,
            message=f"No records have PHASE II label (total: {total_count})",
        )
    else:
        return CheckResult(
            name=f"log_phase_ii_labels_{log_path.name}",
            passed=False,
            message=f"Only {labeled_count}/{total_count} records have PHASE II label",
        )


def verify_ht_series_integrity(
    ht_series_path: Path, manifest: Dict[str, Any]
) -> CheckResult:
    """
    Verify Hₜ series file integrity against manifest declaration.

    Returns CheckResult.
    """
    if not ht_series_path or not ht_series_path.exists():
        return CheckResult(
            name="ht_series_integrity",
            passed=True,
            message="No ht_series.json file (optional, skipped)",
        )

    declared_hash = manifest.get("ht_series_hash")
    if not declared_hash:
        return CheckResult(
            name="ht_series_integrity",
            passed=True,
            message="No ht_series_hash in manifest (file exists but not declared)",
        )

    try:
        with open(ht_series_path, "r", encoding="utf-8") as f:
            ht_data = json.load(f)

        # Compute hash
        import hashlib

        ht_str = json.dumps(ht_data, sort_keys=True)
        actual_hash = hashlib.sha256(ht_str.encode("utf-8")).hexdigest()

        if declared_hash == actual_hash:
            return CheckResult(
                name="ht_series_integrity",
                passed=True,
                message="Hₜ series hash matches manifest declaration",
                expected=declared_hash,
                actual=actual_hash,
            )
        else:
            return CheckResult(
                name="ht_series_integrity",
                passed=False,
                message="Hₜ series hash mismatch",
                expected=declared_hash,
                actual=actual_hash,
            )
    except json.JSONDecodeError as e:
        return CheckResult(
            name="ht_series_integrity",
            passed=False,
            message=f"Invalid JSON in ht_series.json: {e}",
        )
    except Exception as e:
        return CheckResult(
            name="ht_series_integrity",
            passed=False,
            message=f"Error reading ht_series.json: {e}",
        )


def generate_audit_report(
    experiment_dir: Path,
    discovery_notes: List[str],
    manifest_report: Optional[VerificationReport],
    additional_checks: List[CheckResult],
    baseline_counts: Dict[str, int],
    rfl_counts: Dict[str, int],
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate combined Markdown and JSON audit report.

    Returns:
        Tuple of (markdown_report, json_report)
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    overall_pass = True

    # Check manifest report
    if manifest_report and not manifest_report.overall_pass:
        overall_pass = False

    # Check additional checks
    for check in additional_checks:
        if not check.passed:
            overall_pass = False

    # Build JSON report
    json_report: Dict[str, Any] = {
        "audit_version": "1.0.0",
        "experiment_dir": str(experiment_dir),
        "timestamp_utc": timestamp,
        "overall_pass": overall_pass,
        "phase": "II",
        "label": PHASE_II_LABEL,
        "discovery": {
            "notes": discovery_notes,
        },
        "manifest_verification": manifest_report.to_dict() if manifest_report else None,
        "additional_checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "message": c.message,
                "expected": c.expected,
                "actual": c.actual,
            }
            for c in additional_checks
        ],
        "raw_counts": {
            "baseline": baseline_counts,
            "rfl": rfl_counts,
        },
        "interpretation_note": "This report contains raw counts only. No uplift significance testing or interpretation is performed.",
    }

    # Build Markdown report
    md_lines = [
        "# U2 Uplift Experiment Audit Report",
        "",
        "**PHASE II — NOT USED IN PHASE I**",
        "",
        f"**Experiment Directory:** `{experiment_dir}`",
        "",
        f"**Timestamp (UTC):** {timestamp}",
        "",
        f"## Overall Result: {'✅ PASS' if overall_pass else '❌ FAIL'}",
        "",
        "---",
        "",
        "## Artifact Discovery",
        "",
    ]
    for note in discovery_notes:
        md_lines.append(f"- {note}")

    md_lines.extend([
        "",
        "---",
        "",
        "## Manifest Verification",
        "",
    ])

    if manifest_report:
        md_lines.append(f"**Manifest:** `{manifest_report.manifest_path}`")
        md_lines.append("")
        md_lines.append("| Check | Status | Message |")
        md_lines.append("|-------|--------|---------|")
        for check in manifest_report.checks:
            status = "✅ PASS" if check.passed else "❌ FAIL"
            md_lines.append(f"| {check.name} | {status} | {check.message} |")
    else:
        md_lines.append("*No manifest found for verification*")

    md_lines.extend([
        "",
        "---",
        "",
        "## Additional Checks",
        "",
        "| Check | Status | Message |",
        "|-------|--------|---------|",
    ])
    for check in additional_checks:
        status = "✅ PASS" if check.passed else "❌ FAIL"
        md_lines.append(f"| {check.name} | {status} | {check.message} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## Raw Counts (No Interpretation)",
        "",
        "### Baseline Log",
        "",
    ])
    for key, value in baseline_counts.items():
        md_lines.append(f"- **{key}:** {value}")

    md_lines.extend([
        "",
        "### RFL Log",
        "",
    ])
    for key, value in rfl_counts.items():
        md_lines.append(f"- **{key}:** {value}")

    md_lines.extend([
        "",
        "---",
        "",
        "## Notes",
        "",
        "- This report is strictly read-only and does not modify any attestation files.",
        "- **No uplift significance testing or interpretation is performed.**",
        "- Raw counts are provided for audit purposes only.",
        "",
    ])

    return "\n".join(md_lines), json_report


def audit_experiment(experiment_dir: Path) -> int:
    """
    Run complete audit on an experiment directory.

    Args:
        experiment_dir: Path to the experiment directory

    Returns:
        Exit code (0 = pass, 1 = fail, 2 = error)
    """
    print("=" * 70)
    print("U2 UPLIFT EXPERIMENT AUDIT")
    print("PHASE II — NOT USED IN PHASE I")
    print("=" * 70)
    print()
    print(f"Experiment Directory: {experiment_dir}")
    print()

    # 1. Discover artifacts
    print("## Discovering Artifacts...")
    manifest_path, baseline_log, rfl_log, ht_series, discovery_notes = discover_artifacts(
        experiment_dir
    )

    for note in discovery_notes:
        print(f"  {note}")
    print()

    if not experiment_dir.exists():
        print("ERROR: Experiment directory does not exist.", file=sys.stderr)
        return 2

    # 2. Verify manifest if found
    manifest_report: Optional[VerificationReport] = None
    manifest_data: Dict[str, Any] = {}

    if manifest_path:
        print("## Verifying Manifest...")
        verifier = ManifestVerifier(manifest_path, experiment_dir)
        manifest_report = verifier.validate_all(
            baseline_log_path=baseline_log,
            rfl_log_path=rfl_log,
            ht_series_path=ht_series,
        )
        manifest_data = verifier.manifest

        for check in manifest_report.checks:
            status = "✅" if check.passed else "❌"
            print(f"  {status} {check.name}: {check.message}")
        print()
    else:
        print("## Skipping manifest verification (no manifest found)")
        print()

    # 3. Additional checks
    print("## Running Additional Checks...")
    additional_checks: List[CheckResult] = []

    # Check log Phase II labels
    if baseline_log:
        check = check_log_phase_ii_labels(baseline_log)
        additional_checks.append(check)
        status = "✅" if check.passed else "❌"
        print(f"  {status} {check.name}: {check.message}")

    if rfl_log:
        check = check_log_phase_ii_labels(rfl_log)
        additional_checks.append(check)
        status = "✅" if check.passed else "❌"
        print(f"  {status} {check.name}: {check.message}")

    # Check ht_series integrity
    if ht_series:
        check = verify_ht_series_integrity(ht_series, manifest_data)
        additional_checks.append(check)
        status = "✅" if check.passed else "❌"
        print(f"  {status} {check.name}: {check.message}")

    print()

    # 4. Count records (no interpretation)
    print("## Counting Records (No Interpretation)...")
    baseline_counts = count_log_records(baseline_log) if baseline_log else {}
    rfl_counts = count_log_records(rfl_log) if rfl_log else {}

    print(f"  Baseline: {baseline_counts}")
    print(f"  RFL: {rfl_counts}")
    print()

    # 5. Generate reports
    print("## Generating Reports...")
    md_report, json_report = generate_audit_report(
        experiment_dir=experiment_dir,
        discovery_notes=discovery_notes,
        manifest_report=manifest_report,
        additional_checks=additional_checks,
        baseline_counts=baseline_counts,
        rfl_counts=rfl_counts,
    )

    # Write reports
    md_path = experiment_dir / "audit_report.md"
    json_path = experiment_dir / "audit_report.json"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"  ✅ Markdown report: {md_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)
    print(f"  ✅ JSON report: {json_path}")
    print()

    # 6. Summary
    overall_pass = json_report["overall_pass"]
    print("=" * 70)
    if overall_pass:
        print("AUDIT RESULT: ✅ PASS")
    else:
        print("AUDIT RESULT: ❌ FAIL")
    print("=" * 70)
    print()
    print("Note: No uplift significance testing or interpretation was performed.")
    print("      Raw counts are provided for audit purposes only.")
    print()

    return 0 if overall_pass else 1


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="U2 Uplift Experiment Auditor (PHASE II — NOT USED IN PHASE I)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python experiments/audit_uplift_u2.py results/uplift_u2/slice_uplift_goal
    uv run python experiments/audit_uplift_u2.py ./my_experiment/

This auditor:
- Auto-discovers manifest, baseline log, RFL log, and ht_series.json
- Validates all manifest bindings (slice config, prereg, ht_series hashes)
- Validates PHASE II labelling in all artifacts
- Emits Markdown + JSON audit reports
- Does NOT perform uplift significance testing or interpretation
        """,
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to the experiment directory containing manifest and logs",
    )

    args = parser.parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()

    return audit_experiment(experiment_dir)


if __name__ == "__main__":
    sys.exit(main())
