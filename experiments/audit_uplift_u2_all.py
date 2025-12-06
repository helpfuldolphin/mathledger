"""
Multi-experiment audit runner for uplift_u2 Evidence Packs.

Recursively discovers and audits all uplift_u2 experiment directories
in a given tree, aggregating results into consolidated reports.

Exit codes:
- 0: PASS - All sub-audits passed
- 1: FAIL - One or more sub-audits failed (structural/cryptographic errors)
- 2: MIXED - Some experiments missing/ambiguous but others completed

PHASE II — NOT USED IN PHASE I
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from experiments.audit_uplift_u2 import (
    audit_experiment,
    format_audit_report_json,
)


def discover_experiments(root_dir: Path, marker_file: str = "manifest.json") -> List[Path]:
    """
    Recursively discover experiment directories.
    
    Args:
        root_dir: Root directory to search
        marker_file: File that marks an experiment directory (default: manifest.json)
        
    Returns:
        List of paths to experiment directories
    """
    experiment_dirs = []
    
    # Use rglob to find all marker files recursively
    for manifest_path in root_dir.rglob(marker_file):
        experiment_dir = manifest_path.parent
        experiment_dirs.append(experiment_dir)
    
    return sorted(experiment_dirs)


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate multiple audit results into summary statistics.
    
    Args:
        results: List of audit result dictionaries
        
    Returns:
        Aggregated statistics dictionary
    """
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    missing = sum(1 for r in results if r["status"] == "MISSING")
    
    # Collect all findings by category
    findings_by_category: Dict[str, int] = {}
    for result in results:
        for finding in result.get("findings", []):
            category = finding.get("category", "unknown")
            findings_by_category[category] = findings_by_category.get(category, 0) + 1
    
    return {
        "total_experiments": total,
        "passed": passed,
        "failed": failed,
        "missing": missing,
        "findings_by_category": findings_by_category,
    }


def format_multi_audit_json(
    root_dir: Path,
    experiment_results: List[Dict[str, Any]],
    summary: Dict[str, Any]
) -> Dict[str, Any]:
    """Format multi-audit results as JSON."""
    return {
        "root_dir": str(root_dir),
        "summary": summary,
        "experiments": experiment_results,
    }


def format_multi_audit_markdown(
    root_dir: Path,
    experiment_results: List[Dict[str, Any]],
    summary: Dict[str, Any]
) -> str:
    """Format multi-audit results as Markdown."""
    lines = []
    
    lines.append("# Multi-Experiment Audit Report")
    lines.append("")
    lines.append(f"**Root Directory**: `{root_dir}`")
    lines.append("")
    
    # Summary section
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total Experiments**: {summary['total_experiments']}")
    lines.append(f"- **Passed**: {summary['passed']} ✅")
    lines.append(f"- **Failed**: {summary['failed']} ❌")
    lines.append(f"- **Missing/Incomplete**: {summary['missing']} ⚠️")
    lines.append("")
    
    # Overall status
    if summary['failed'] > 0:
        overall_status = "**FAIL** ❌"
        lines.append(f"**Overall Status**: {overall_status}")
        lines.append("")
        lines.append("One or more experiments have structural or cryptographic failures.")
    elif summary['missing'] > 0:
        overall_status = "**MIXED** ⚠️"
        lines.append(f"**Overall Status**: {overall_status}")
        lines.append("")
        lines.append("Some experiments are missing required artifacts, but no failures detected.")
    else:
        overall_status = "**PASS** ✅"
        lines.append(f"**Overall Status**: {overall_status}")
        lines.append("")
        lines.append("All experiments passed integrity checks.")
    
    lines.append("")
    
    # Findings by category
    if summary['findings_by_category']:
        lines.append("## Findings by Category")
        lines.append("")
        lines.append("| Category | Count |")
        lines.append("|----------|-------|")
        for category, count in sorted(summary['findings_by_category'].items()):
            lines.append(f"| {category} | {count} |")
        lines.append("")
    
    # Per-experiment details
    lines.append("## Experiment Details")
    lines.append("")
    
    for i, result in enumerate(experiment_results, 1):
        exp_dir = result["experiment_dir"]
        status = result["status"]
        findings_count = len(result.get("findings", []))
        
        # Status emoji
        if status == "PASS":
            status_emoji = "✅"
        elif status == "FAIL":
            status_emoji = "❌"
        else:
            status_emoji = "⚠️"
        
        lines.append(f"### {i}. `{exp_dir}` {status_emoji}")
        lines.append(f"- **Status**: {status}")
        lines.append(f"- **Findings**: {findings_count}")
        
        # Show findings if any
        if findings_count > 0:
            lines.append("- **Issues**:")
            for finding in result["findings"]:
                severity = finding["severity"]
                message = finding["message"]
                lines.append(f"  - [{severity}] {message}")
        
        lines.append("")
    
    return "\n".join(lines)


def audit_all_experiments(
    root_dir: Path,
    quiet: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Discover and audit all experiments under root_dir.
    
    Args:
        root_dir: Root directory to search
        quiet: Suppress progress output
        
    Returns:
        Tuple of (list of experiment results, aggregated summary)
    """
    if not quiet:
        print(f"Discovering experiments in: {root_dir}")
    
    experiment_dirs = discover_experiments(root_dir)
    
    if not quiet:
        print(f"Found {len(experiment_dirs)} experiment(s)")
        print()
    
    results = []
    
    for i, exp_dir in enumerate(experiment_dirs, 1):
        if not quiet:
            print(f"[{i}/{len(experiment_dirs)}] Auditing: {exp_dir}")
        
        result = audit_experiment(exp_dir)
        result_dict = format_audit_report_json(result)
        results.append(result_dict)
        
        if not quiet:
            status = result_dict["status"]
            findings_count = len(result_dict.get("findings", []))
            print(f"    Status: {status}, Findings: {findings_count}")
    
    if not quiet:
        print()
    
    summary = aggregate_results(results)
    
    return results, summary


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Audit all uplift_u2 experiments in a directory tree.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Exit Codes:
  0 - PASS: All sub-audits passed
  1 - FAIL: One or more sub-audits failed (structural/cryptographic errors)
  2 - MIXED: Some experiments missing/ambiguous but others completed

Examples:
  # Audit all experiments under results/uplift_u2
  python experiments/audit_uplift_u2_all.py results/uplift_u2

  # Generate reports
  python experiments/audit_uplift_u2_all.py results/uplift_u2 \\
    --output-json multi_audit.json \\
    --output-md multi_audit.md

Usage Notes:
  - This script recursively searches for manifest.json files
  - Each directory containing manifest.json is treated as an experiment
  - Results are aggregated across all discovered experiments
  - Use --quiet to suppress progress output and only see final results
        """
    )
    
    parser.add_argument(
        "root_dir",
        type=str,
        default="results/uplift_u2",
        nargs='?',
        help="Root directory to search for experiments (default: results/uplift_u2)"
    )
    
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write aggregated JSON report to file"
    )
    
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Write aggregated Markdown report to file"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir).resolve()
    
    if not root_dir.exists():
        print(f"ERROR: Root directory does not exist: {root_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not root_dir.is_dir():
        print(f"ERROR: Root path is not a directory: {root_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Run multi-experiment audit
    experiment_results, summary = audit_all_experiments(root_dir, quiet=args.quiet)
    
    # Generate reports
    json_report = format_multi_audit_json(root_dir, experiment_results, summary)
    md_report = format_multi_audit_markdown(root_dir, experiment_results, summary)
    
    # Output JSON report
    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2)
        if not args.quiet:
            print(f"JSON report written to: {output_json_path}")
    
    # Output Markdown report
    if args.output_md:
        output_md_path = Path(args.output_md)
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        if not args.quiet:
            print(f"Markdown report written to: {output_md_path}")
    
    # Print summary to stdout
    if not args.quiet or (not args.output_json and not args.output_md):
        print("=" * 60)
        print("MULTI-AUDIT SUMMARY")
        print("=" * 60)
        print(f"Total Experiments: {summary['total_experiments']}")
        print(f"Passed:            {summary['passed']}")
        print(f"Failed:            {summary['failed']}")
        print(f"Missing:           {summary['missing']}")
        print("=" * 60)
    
    # Determine exit code based on aggregated results
    if summary['failed'] > 0:
        if not args.quiet:
            print("Overall Status: FAIL (one or more experiments failed)")
        sys.exit(1)
    elif summary['missing'] > 0:
        if not args.quiet:
            print("Overall Status: MIXED (some experiments have missing artifacts)")
        sys.exit(2)
    else:
        if not args.quiet:
            print("Overall Status: PASS (all experiments passed)")
        sys.exit(0)


if __name__ == "__main__":
    main()
