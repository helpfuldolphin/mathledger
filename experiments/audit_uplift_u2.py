#!/usr/bin/env python3
"""
Phase II U2 Uplift Experiment Auditor

This script audits Phase II U2 uplift experiments for structural and
cryptographic integrity. It does NOT compute uplift or p-values; it only
verifies that artifacts are internally and externally consistent.

Exit Codes:
    0 - PASS: All checks OK
    1 - FAIL: Structural/cryptographic failure detected
    2 - MISSING: Missing or ambiguous artifacts

Usage:
    python experiments/audit_uplift_u2.py --experiment-dir <path>
    python experiments/audit_uplift_u2.py --manifest <path>

Outputs:
    - audit_report.json: Machine-readable audit results
    - audit_report.md: Human-readable audit report

SOBER TRUTH GUARDRAILS:
    - Do NOT modify manifests or logs; this is a read-only auditor
    - Do NOT compute uplift or p-values
    - Do NOT interpret audit findings as uplift evidence
    - Keep everything Phase II only; do not touch Phase I attestation paths

Phase II only — NOT USED IN PHASE I.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Handle both direct execution and module import
try:
    from experiments.manifest_verifier import ManifestVerifier, ManifestVerificationReport
except ImportError:
    from manifest_verifier import ManifestVerifier, ManifestVerificationReport


# Exit codes
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_MISSING = 2


def compute_file_hash(filepath: Path) -> Optional[str]:
    """Compute SHA256 hash of a file."""
    if not filepath.exists():
        return None
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except (IOError, OSError):
        return None


def count_jsonl_records(filepath: Path) -> Optional[int]:
    """Count valid JSON records in a JSONL file."""
    if not filepath.exists():
        return None
    count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                        count += 1
                    except json.JSONDecodeError:
                        pass
    except (IOError, OSError):
        return None
    return count


def discover_artifacts(experiment_dir: Path) -> Dict[str, Any]:
    """
    Discover manifest, logs, and ht_series in an experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        Dictionary with discovered artifact paths and metadata
    """
    artifacts = {
        "experiment_dir": str(experiment_dir),
        "manifest": None,
        "baseline_log": None,
        "rfl_log": None,
        "ht_series": None,
        "discovered_files": []
    }
    
    if not experiment_dir.exists():
        return artifacts
    
    # Look for manifest files
    manifest_patterns = [
        "manifest.json",
        "*_manifest_*.json",
        "uplift_u2_manifest_*.json"
    ]
    
    for pattern in manifest_patterns:
        matches = list(experiment_dir.glob(pattern))
        if matches:
            artifacts["manifest"] = str(matches[0])
            break
    
    # Look for JSONL log files
    for jsonl_file in experiment_dir.glob("*.jsonl"):
        artifacts["discovered_files"].append(str(jsonl_file))
        filename_lower = jsonl_file.name.lower()
        if "baseline" in filename_lower:
            artifacts["baseline_log"] = str(jsonl_file)
        elif "rfl" in filename_lower:
            artifacts["rfl_log"] = str(jsonl_file)
    
    # Look for ht_series.json
    ht_series_path = experiment_dir / "ht_series.json"
    if ht_series_path.exists():
        artifacts["ht_series"] = str(ht_series_path)
    
    return artifacts


def validate_log_structure(log_path: Path) -> Dict[str, Any]:
    """
    Validate the structure of a JSONL log file.
    
    Args:
        log_path: Path to the JSONL log file
        
    Returns:
        Validation result dictionary
    """
    result = {
        "path": str(log_path),
        "exists": log_path.exists(),
        "record_count": None,
        "has_phase_ii_label": False,
        "has_cycle_field": False,
        "is_valid": False,
        "issues": []
    }
    
    if not log_path.exists():
        result["issues"].append("Log file does not exist")
        return result
    
    if log_path.stat().st_size == 0:
        result["issues"].append("Log file is empty (0 bytes)")
        return result
    
    records = []
    parse_errors = 0
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    parse_errors += 1
    except (IOError, OSError) as e:
        result["issues"].append(f"Error reading file: {e}")
        return result
    
    result["record_count"] = len(records)
    
    if parse_errors > 0:
        result["issues"].append(f"{parse_errors} lines failed to parse as JSON")
    
    if not records:
        result["issues"].append("No valid JSON records found")
        return result
    
    # Check for Phase II label
    sample = records[0]
    label = sample.get("label", "")
    if "PHASE II" in label or "phase_ii" in label.lower():
        result["has_phase_ii_label"] = True
    else:
        result["issues"].append("Missing Phase II label in log records")
    
    # Check for cycle field
    if "cycle" in sample:
        result["has_cycle_field"] = True
    else:
        result["issues"].append("Missing 'cycle' field in log records")
    
    # Check cycle continuity
    if result["has_cycle_field"]:
        cycles = [r.get("cycle") for r in records if "cycle" in r]
        if cycles:
            expected = set(range(len(cycles)))
            actual = set(cycles)
            if expected != actual:
                missing = sorted(expected - actual)
                duplicate = sorted([c for c in cycles if cycles.count(c) > 1])
                if missing:
                    result["issues"].append(f"Missing cycle indices: {missing[:10]}...")
                if duplicate:
                    result["issues"].append(f"Duplicate cycle indices: {list(set(duplicate))[:10]}...")
    
    result["is_valid"] = len(result["issues"]) == 0
    return result


def generate_audit_report(
    experiment_dir: Path,
    manifest_report: Optional[ManifestVerificationReport],
    artifacts: Dict[str, Any],
    log_validations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate comprehensive audit report.
    
    Args:
        experiment_dir: Path to experiment directory
        manifest_report: Manifest verification report (if available)
        artifacts: Discovered artifacts
        log_validations: Log validation results
        
    Returns:
        Complete audit report dictionary
    """
    report = {
        "meta": {
            "auditor": "audit_uplift_u2",
            "version": "1.0.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "phase": "II",
            "label": "PHASE II — NOT USED IN PHASE I"
        },
        "experiment_dir": str(experiment_dir),
        "artifacts": artifacts,
        "manifest_verification": manifest_report.to_dict() if manifest_report else None,
        "log_validations": log_validations,
        "summary": {
            "overall_status": "unknown",
            "checks_passed": 0,
            "checks_failed": 0,
            "checks_skipped": 0,
            "issues": [],
            "warnings": []
        }
    }
    
    # Tally results
    all_issues = []
    
    if manifest_report:
        for result in manifest_report.results:
            if result.passed:
                report["summary"]["checks_passed"] += 1
            elif result.details.get("skipped"):
                report["summary"]["checks_skipped"] += 1
            else:
                report["summary"]["checks_failed"] += 1
                all_issues.append(f"[manifest] {result.check_name}: {result.message}")
    else:
        report["summary"]["checks_failed"] += 1
        all_issues.append("[manifest] Manifest not found or could not be verified")
    
    for log_val in log_validations:
        if log_val["is_valid"]:
            report["summary"]["checks_passed"] += 1
        else:
            report["summary"]["checks_failed"] += 1
            for issue in log_val["issues"]:
                all_issues.append(f"[log:{Path(log_val['path']).name}] {issue}")
    
    report["summary"]["issues"] = all_issues
    
    # Determine overall status
    if report["summary"]["checks_failed"] == 0:
        report["summary"]["overall_status"] = "PASS"
    elif any("not found" in i.lower() or "not exist" in i.lower() or "missing" in i.lower() for i in all_issues):
        report["summary"]["overall_status"] = "MISSING"
    else:
        report["summary"]["overall_status"] = "FAIL"
    
    return report


def generate_markdown_report(audit_report: Dict[str, Any]) -> str:
    """
    Generate human-readable Markdown report.
    
    Args:
        audit_report: Audit report dictionary
        
    Returns:
        Markdown formatted report string
    """
    lines = []
    
    # Header with exit code documentation
    lines.append("# Phase II U2 Uplift Experiment Audit Report")
    lines.append("")
    lines.append(f"**Generated**: {audit_report['meta']['timestamp_utc']}")
    lines.append(f"**Auditor Version**: {audit_report['meta']['version']}")
    lines.append(f"**Phase**: {audit_report['meta']['phase']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Exit Code Reference")
    lines.append("")
    lines.append("| Code | Status | Description |")
    lines.append("|------|--------|-------------|")
    lines.append("| 0 | PASS | All checks OK |")
    lines.append("| 1 | FAIL | Structural/cryptographic failure |")
    lines.append("| 2 | MISSING | Missing or ambiguous artifacts |")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Summary
    summary = audit_report["summary"]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"**Overall Status**: `{summary['overall_status']}`")
    lines.append("")
    lines.append(f"- Checks Passed: {summary['checks_passed']}")
    lines.append(f"- Checks Failed: {summary['checks_failed']}")
    lines.append(f"- Checks Skipped: {summary['checks_skipped']}")
    lines.append("")
    
    if summary["issues"]:
        lines.append("### Issues")
        lines.append("")
        for issue in summary["issues"]:
            lines.append(f"- ❌ {issue}")
        lines.append("")
    
    if summary["warnings"]:
        lines.append("### Warnings")
        lines.append("")
        for warning in summary["warnings"]:
            lines.append(f"- ⚠️ {warning}")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Discovered Artifacts
    artifacts = audit_report["artifacts"]
    lines.append("## Discovered Artifacts")
    lines.append("")
    lines.append(f"**Experiment Directory**: `{artifacts['experiment_dir']}`")
    lines.append("")
    lines.append("| Artifact | Path |")
    lines.append("|----------|------|")
    lines.append(f"| Manifest | {artifacts.get('manifest') or 'Not found'} |")
    lines.append(f"| Baseline Log | {artifacts.get('baseline_log') or 'Not found'} |")
    lines.append(f"| RFL Log | {artifacts.get('rfl_log') or 'Not found'} |")
    lines.append(f"| ht_series | {artifacts.get('ht_series') or 'Not found'} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Manifest Verification
    manifest_ver = audit_report.get("manifest_verification")
    lines.append("## Manifest Verification")
    lines.append("")
    
    if manifest_ver:
        lines.append(f"**Status**: `{manifest_ver['overall_status']}`")
        lines.append(f"**Path**: `{manifest_ver['manifest_path']}`")
        lines.append("")
        lines.append("### Check Results")
        lines.append("")
        lines.append("| Check | Status | Message |")
        lines.append("|-------|--------|---------|")
        for result in manifest_ver["results"]:
            status = "✅" if result["passed"] else "❌"
            lines.append(f"| {result['check_name']} | {status} | {result['message']} |")
        lines.append("")
    else:
        lines.append("❌ Manifest could not be verified.")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Log Validations
    log_vals = audit_report.get("log_validations", [])
    lines.append("## Log Validations")
    lines.append("")
    
    if log_vals:
        for log_val in log_vals:
            status = "✅" if log_val["is_valid"] else "❌"
            lines.append(f"### {Path(log_val['path']).name}")
            lines.append("")
            lines.append(f"- **Status**: {status}")
            lines.append(f"- **Path**: `{log_val['path']}`")
            lines.append(f"- **Exists**: {log_val['exists']}")
            lines.append(f"- **Record Count**: {log_val['record_count']}")
            lines.append(f"- **Has Phase II Label**: {log_val['has_phase_ii_label']}")
            lines.append(f"- **Has Cycle Field**: {log_val['has_cycle_field']}")
            if log_val["issues"]:
                lines.append("- **Issues**:")
                for issue in log_val["issues"]:
                    lines.append(f"  - {issue}")
            lines.append("")
    else:
        lines.append("No logs found to validate.")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("*This is a read-only audit report. No files were modified.*")
    lines.append("")
    lines.append("**PHASE II — NOT USED IN PHASE I**")
    
    return "\n".join(lines)


def audit_experiment(
    experiment_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> int:
    """
    Audit a Phase II U2 uplift experiment.
    
    Args:
        experiment_dir: Path to experiment directory
        manifest_path: Path to manifest file (alternative to experiment_dir)
        output_dir: Directory for output reports (defaults to experiment_dir)
        
    Returns:
        Exit code (0=PASS, 1=FAIL, 2=MISSING)
    """
    # Resolve experiment directory
    if manifest_path:
        manifest_path = Path(manifest_path)
        experiment_dir = manifest_path.parent
    elif experiment_dir:
        experiment_dir = Path(experiment_dir)
    else:
        print("ERROR: Must specify --experiment-dir or --manifest", file=sys.stderr)
        return EXIT_MISSING
    
    if output_dir is None:
        output_dir = experiment_dir
    else:
        output_dir = Path(output_dir)
    
    print(f"Auditing experiment: {experiment_dir}")
    print(f"PHASE II — NOT USED IN PHASE I")
    print("")
    
    # Discover artifacts
    artifacts = discover_artifacts(experiment_dir)
    
    # Find and verify manifest
    manifest_report = None
    if artifacts["manifest"]:
        manifest_path = Path(artifacts["manifest"])
        verifier = ManifestVerifier(manifest_path, experiment_dir)
        manifest_report = verifier.verify_all()
        print(f"Manifest verification: {manifest_report.overall_status}")
    else:
        print("WARNING: No manifest found in experiment directory")
    
    # Validate logs
    log_validations = []
    
    if artifacts["baseline_log"]:
        baseline_val = validate_log_structure(Path(artifacts["baseline_log"]))
        log_validations.append(baseline_val)
        status = "✓" if baseline_val["is_valid"] else "✗"
        print(f"Baseline log validation: {status} ({baseline_val['record_count']} records)")
    
    if artifacts["rfl_log"]:
        rfl_val = validate_log_structure(Path(artifacts["rfl_log"]))
        log_validations.append(rfl_val)
        status = "✓" if rfl_val["is_valid"] else "✗"
        print(f"RFL log validation: {status} ({rfl_val['record_count']} records)")
    
    # Generate audit report
    audit_report = generate_audit_report(
        experiment_dir,
        manifest_report,
        artifacts,
        log_validations
    )
    
    # Write JSON report
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "audit_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(audit_report, f, indent=2)
    print(f"JSON report: {json_path}")
    
    # Write Markdown report
    md_path = output_dir / "audit_report.md"
    md_content = generate_markdown_report(audit_report)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Markdown report: {md_path}")
    
    # Print summary
    print("")
    print(f"Overall status: {audit_report['summary']['overall_status']}")
    print(f"Checks passed: {audit_report['summary']['checks_passed']}")
    print(f"Checks failed: {audit_report['summary']['checks_failed']}")
    
    if audit_report["summary"]["issues"]:
        print("")
        print("Issues:")
        for issue in audit_report["summary"]["issues"]:
            print(f"  - {issue}")
    
    # Return appropriate exit code
    status = audit_report["summary"]["overall_status"]
    if status == "PASS":
        return EXIT_PASS
    elif status == "MISSING":
        return EXIT_MISSING
    else:
        return EXIT_FAIL


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="""
Phase II U2 Uplift Experiment Auditor

Exit Codes:
    0 - PASS: All checks OK
    1 - FAIL: Structural/cryptographic failure
    2 - MISSING: Missing or ambiguous artifacts

This is a read-only auditor. It does NOT compute uplift or modify any files.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--experiment-dir",
        type=str,
        help="Path to the experiment directory"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="Path to manifest file (alternative to --experiment-dir)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for output reports (defaults to experiment directory)"
    )
    
    args = parser.parse_args()
    
    if not args.experiment_dir and not args.manifest:
        parser.error("Must specify --experiment-dir or --manifest")
    
    exit_code = audit_experiment(
        experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        manifest_path=Path(args.manifest) if args.manifest else None,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
