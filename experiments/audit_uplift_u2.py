"""
Audit uplift_u2 Evidence Packs for integrity and consistency.

This script verifies:
- Manifest exists and is valid JSON
- Log files (baseline.jsonl, rfl.jsonl) exist and are non-empty
- Cycle counts in logs match manifest declarations
- ht_series.json exists if referenced in manifest
- All artifact hashes are computed for tamper-evident reporting

Exit codes:
- 0: PASS - All checks passed
- 1: FAIL - Structural/cryptographic failures detected
- 2: MISSING - Required artifacts missing or incomplete

PHASE II — NOT USED IN PHASE I
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from experiments.manifest_verifier import compute_artifact_hash, hash_string


class AuditResult:
    """Container for audit findings."""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.findings: List[Dict[str, Any]] = []
        self.artifact_hashes: Dict[str, str] = {}
        self.status = "PASS"  # PASS, FAIL, MISSING
        
    def add_finding(self, severity: str, category: str, message: str, detail: Optional[str] = None):
        """Add an audit finding."""
        finding = {
            "severity": severity,
            "category": category,
            "message": message,
        }
        if detail:
            finding["detail"] = detail
        self.findings.append(finding)
        
        # Update status based on severity
        if severity == "ERROR":
            if self.status == "PASS":
                self.status = "FAIL"
        elif severity == "MISSING":
            if self.status == "PASS":
                self.status = "MISSING"
    
    def add_artifact_hash(self, artifact_path: Path, relative_to: Optional[Path] = None):
        """Compute and store hash for an artifact."""
        # Make path relative for cleaner reporting
        if relative_to and artifact_path.is_absolute():
            try:
                display_path = artifact_path.relative_to(relative_to)
            except ValueError:
                display_path = artifact_path
        else:
            display_path = artifact_path
        
        hash_value = compute_artifact_hash(artifact_path)
        self.artifact_hashes[str(display_path)] = hash_value
        return hash_value


def count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    if not path.exists():
        return 0
    
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def audit_experiment(experiment_dir: Path) -> AuditResult:
    """
    Audit a single uplift_u2 experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory containing manifest.json
        
    Returns:
        AuditResult with findings and artifact hashes
    """
    result = AuditResult(experiment_dir)
    
    # Check 1: Manifest exists
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.exists():
        result.add_finding("MISSING", "manifest", 
                          f"Manifest file not found at {manifest_path}")
        return result
    
    # Compute manifest hash
    result.add_artifact_hash(manifest_path, relative_to=experiment_dir)
    
    # Check 2: Manifest is valid JSON
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        result.add_finding("ERROR", "manifest", 
                          "Manifest is not valid JSON", detail=str(e))
        return result
    
    # Check 3: Required manifest fields
    required_fields = ["slice", "mode", "cycles"]
    for field in required_fields:
        if field not in manifest:
            result.add_finding("ERROR", "manifest", 
                              f"Required field '{field}' missing from manifest")
    
    # Extract manifest metadata
    slice_name = manifest.get("slice", "unknown")
    mode = manifest.get("mode", "unknown")
    declared_cycles = manifest.get("cycles", 0)
    
    # Check 4: Baseline log file
    outputs = manifest.get("outputs", {})
    baseline_log_path = None
    rfl_log_path = None
    
    # Try to find log files from manifest or by convention
    if "baseline_results" in outputs:
        baseline_log_path = experiment_dir / outputs["baseline_results"]
    elif "results" in outputs:
        results_path = outputs["results"]
        if "baseline" in results_path:
            baseline_log_path = experiment_dir / results_path
    
    # Fallback to convention
    if not baseline_log_path:
        baseline_log_path = experiment_dir / f"uplift_u2_{slice_name}_baseline.jsonl"
    
    if not baseline_log_path.exists():
        result.add_finding("MISSING", "log_file", 
                          f"Baseline log file not found: {baseline_log_path.name}")
    else:
        baseline_hash = result.add_artifact_hash(baseline_log_path, relative_to=experiment_dir)
        
        # Check for empty file
        baseline_count = count_jsonl_lines(baseline_log_path)
        if baseline_count == 0:
            result.add_finding("ERROR", "log_file", 
                              f"Baseline log file is empty: {baseline_log_path.name}")
        else:
            # Verify cycle count matches manifest
            if declared_cycles > 0 and baseline_count != declared_cycles:
                result.add_finding("ERROR", "cycle_count", 
                                  f"Baseline cycle count mismatch: manifest={declared_cycles}, log={baseline_count}")
    
    # Check 5: RFL log file (if mode is rfl or paired experiment)
    if mode == "rfl" or manifest.get("paired_experiment"):
        if "rfl_results" in outputs:
            rfl_log_path = experiment_dir / outputs["rfl_results"]
        elif not rfl_log_path:
            rfl_log_path = experiment_dir / f"uplift_u2_{slice_name}_rfl.jsonl"
        
        if not rfl_log_path.exists():
            result.add_finding("MISSING", "log_file", 
                              f"RFL log file not found: {rfl_log_path.name}")
        else:
            rfl_hash = result.add_artifact_hash(rfl_log_path, relative_to=experiment_dir)
            
            # Check for empty file
            rfl_count = count_jsonl_lines(rfl_log_path)
            if rfl_count == 0:
                result.add_finding("ERROR", "log_file", 
                                  f"RFL log file is empty: {rfl_log_path.name}")
            else:
                # Verify cycle count matches manifest
                if declared_cycles > 0 and rfl_count != declared_cycles:
                    result.add_finding("ERROR", "cycle_count", 
                                      f"RFL cycle count mismatch: manifest={declared_cycles}, log={rfl_count}")
    
    # Check 6: ht_series hash verification
    if "ht_series_hash" in manifest:
        declared_ht_hash = manifest["ht_series_hash"]
        
        # Try to find ht_series.json
        ht_series_path = experiment_dir / "ht_series.json"
        if ht_series_path.exists():
            ht_hash = result.add_artifact_hash(ht_series_path, relative_to=experiment_dir)
            
            # Verify hash by recomputing from content
            try:
                with open(ht_series_path, 'r', encoding='utf-8') as f:
                    ht_content = json.load(f)
                    computed_hash = hash_string(json.dumps(ht_content, sort_keys=True))
                    
                if computed_hash != declared_ht_hash:
                    result.add_finding("ERROR", "hash_mismatch", 
                                      f"ht_series hash mismatch: manifest={declared_ht_hash[:16]}..., computed={computed_hash[:16]}...")
            except (json.JSONDecodeError, IOError) as e:
                result.add_finding("ERROR", "ht_series", 
                                  f"Failed to verify ht_series.json: {e}")
        else:
            result.add_finding("MISSING", "ht_series", 
                              "ht_series.json referenced in manifest but not found")
    
    # Check 7: Preregistration hash (if present)
    if "prereg_hash" in manifest and manifest["prereg_hash"] != "N/A":
        # This is informational - we don't have access to the original prereg here
        # but we note it for the audit trail
        pass
    
    # Check 8: Look for calibration outputs if mentioned
    if "calibration" in manifest or "calibration_outputs" in outputs:
        calibration_paths = []
        if "calibration_outputs" in outputs:
            calibration_paths = [experiment_dir / p for p in outputs["calibration_outputs"]]
        
        for cal_path in calibration_paths:
            if cal_path.exists():
                result.add_artifact_hash(cal_path, relative_to=experiment_dir)
            else:
                result.add_finding("MISSING", "calibration", 
                                  f"Calibration output not found: {cal_path.name}")
    
    return result


def format_audit_report_json(result: AuditResult) -> Dict[str, Any]:
    """Format audit result as JSON report."""
    return {
        "experiment_dir": str(result.experiment_dir),
        "status": result.status,
        "findings": result.findings,
        "artifact_hashes": result.artifact_hashes,
    }


def format_audit_report_markdown(result: AuditResult) -> str:
    """Format audit result as Markdown report."""
    lines = []
    lines.append("# Uplift U2 Experiment Audit Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- **Experiment Directory**: `{result.experiment_dir}`")
    lines.append(f"- **Status**: **{result.status}**")
    lines.append(f"- **Findings**: {len(result.findings)}")
    lines.append("")
    
    # Artifact Hashes Section
    lines.append("## Artifact Hashes (SHA-256)")
    lines.append("")
    lines.append("These cryptographic hashes provide tamper-evident verification of all")
    lines.append("key artifacts in this Evidence Pack. Each hash is computed from the")
    lines.append("file's binary content using SHA-256.")
    lines.append("")
    
    if result.artifact_hashes:
        lines.append("| Artifact | Hash |")
        lines.append("|----------|------|")
        for artifact_path, hash_value in sorted(result.artifact_hashes.items()):
            lines.append(f"| `{artifact_path}` | `{hash_value}` |")
    else:
        lines.append("*No artifacts found or hashed.*")
    
    lines.append("")
    
    # Findings Section
    lines.append("## Findings")
    lines.append("")
    
    if result.findings:
        for i, finding in enumerate(result.findings, 1):
            severity = finding["severity"]
            category = finding["category"]
            message = finding["message"]
            
            lines.append(f"### Finding {i}: {severity}")
            lines.append(f"- **Category**: {category}")
            lines.append(f"- **Message**: {message}")
            
            if "detail" in finding:
                lines.append(f"- **Detail**: `{finding['detail']}`")
            
            lines.append("")
    else:
        lines.append("✅ No issues found. All integrity checks passed.")
        lines.append("")
    
    return "\n".join(lines)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Audit uplift_u2 Evidence Pack for integrity and consistency.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Exit Codes:
  0 - PASS: All checks passed
  1 - FAIL: Structural or cryptographic failures detected
  2 - MISSING: Required artifacts missing or incomplete

Examples:
  python experiments/audit_uplift_u2.py results/uplift_u2/EXP_001
  python experiments/audit_uplift_u2.py results/uplift_u2/EXP_001 --output-json report.json
        """
    )
    
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory containing manifest.json"
    )
    
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write JSON audit report to file (default: print to stdout)"
    )
    
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Write Markdown audit report to file"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output, only show final status"
    )
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir).resolve()
    
    if not args.quiet:
        print(f"Auditing experiment: {experiment_dir}")
        print()
    
    # Run audit
    result = audit_experiment(experiment_dir)
    
    # Generate reports
    json_report = format_audit_report_json(result)
    md_report = format_audit_report_markdown(result)
    
    # Output JSON report
    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2)
        if not args.quiet:
            print(f"JSON report written to: {output_json_path}")
    else:
        if not args.quiet:
            print("=== JSON Report ===")
        print(json.dumps(json_report, indent=2))
    
    # Output Markdown report
    if args.output_md:
        output_md_path = Path(args.output_md)
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        if not args.quiet:
            print(f"Markdown report written to: {output_md_path}")
    elif not args.output_json:
        # Only print markdown if we didn't print JSON
        if not args.quiet:
            print()
            print("=== Markdown Report ===")
        print(md_report)
    
    # Print final status
    if not args.quiet:
        print()
        print(f"Audit Status: {result.status}")
    
    # Exit with appropriate code
    if result.status == "PASS":
        sys.exit(0)
    elif result.status == "MISSING":
        sys.exit(2)
    else:  # FAIL
        sys.exit(1)


if __name__ == "__main__":
    main()
