"""
Multi-Experiment Auditor for U2 Uplift Experiments
===================================================

Sweeps multiple experiment directories and aggregates audit results.
Provides batch auditing capabilities for CI/CD pipelines.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from attestation.audit_uplift_u2 import audit_experiment, AuditResult


def audit_all_experiments(
    experiments_dir: Path,
    repo_root: Optional[Path] = None,
    experiment_pattern: str = "*",
    prereg_hashes: Optional[Dict[str, str]] = None
) -> List[AuditResult]:
    """
    Audit all experiment directories matching a pattern.
    
    Args:
        experiments_dir: Directory containing experiment subdirectories
        repo_root: Repository root (defaults to experiments_dir parent)
        experiment_pattern: Glob pattern for experiment directories
        prereg_hashes: Optional dict mapping experiment_id to preregistration hash
        
    Returns:
        List of AuditResult objects, one per experiment
    """
    if repo_root is None:
        repo_root = experiments_dir.parent
    
    if prereg_hashes is None:
        prereg_hashes = {}
    
    results = []
    
    # Find all experiment directories
    for exp_dir in sorted(experiments_dir.glob(experiment_pattern)):
        if not exp_dir.is_dir():
            continue
        
        # Skip if no manifest exists
        if not (exp_dir / "manifest.json").exists():
            continue
        
        experiment_id = exp_dir.name
        prereg_hash = prereg_hashes.get(experiment_id)
        
        result = audit_experiment(exp_dir, repo_root, prereg_hash)
        results.append(result)
    
    return results


def aggregate_audit_summary(results: List[AuditResult]) -> Dict[str, Any]:
    """
    Aggregate summary statistics from multiple audit results.
    
    Args:
        results: List of AuditResult objects
        
    Returns:
        Dictionary with aggregate statistics
    """
    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")
    
    total_issues = sum(len(r.issues) for r in results)
    total_artifacts = sum(r.summary.get("total_artifacts", 0) for r in results)
    
    return {
        "total_experiments": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total_issues": total_issues,
        "total_artifacts": total_artifacts,
        "pass_rate": passed / total if total > 0 else 0.0
    }


def render_aggregate_json(results: List[AuditResult]) -> str:
    """
    Render aggregate audit results as JSON.
    
    Args:
        results: List of AuditResult objects
        
    Returns:
        JSON string with all results and summary
    """
    summary = aggregate_audit_summary(results)
    
    experiments = [
        {
            "experiment_id": r.experiment_id,
            "status": r.status,
            "manifest_path": r.manifest_path,
            "manifest_hash": r.manifest_hash,
            "issue_count": len(r.issues),
            "artifact_count": len(r.artifacts_checked)
        }
        for r in results
    ]
    
    data = {
        "summary": summary,
        "experiments": experiments
    }
    
    return json.dumps(data, indent=2)


def render_aggregate_markdown(results: List[AuditResult]) -> str:
    """
    Render aggregate audit results as Markdown report.
    
    Args:
        results: List of AuditResult objects
        
    Returns:
        Markdown string with summary table and statistics
    """
    summary = aggregate_audit_summary(results)
    
    lines = [
        "# Multi-Experiment Audit Report",
        "",
        "## Summary",
        "",
        f"- **Total Experiments:** {summary['total_experiments']}",
        f"- **Passed:** {summary['passed']}",
        f"- **Failed:** {summary['failed']}",
        f"- **Skipped:** {summary['skipped']}",
        f"- **Pass Rate:** {summary['pass_rate']:.1%}",
        f"- **Total Issues:** {summary['total_issues']}",
        f"- **Total Artifacts:** {summary['total_artifacts']}",
        "",
        "## Experiments",
        "",
        "| Experiment ID | Status | Manifest Hash | Issues | Artifacts |",
        "|---------------|--------|---------------|--------|-----------|",
    ]
    
    for result in results:
        status_icon = "✓" if result.status == "PASS" else ("✗" if result.status == "FAIL" else "—")
        manifest_hash_short = result.manifest_hash[:8] if result.manifest_hash else "N/A"
        
        lines.append(
            f"| {result.experiment_id} | {status_icon} {result.status} | "
            f"`{manifest_hash_short}...` | {len(result.issues)} | "
            f"{len(result.artifacts_checked)} |"
        )
    
    # Add failed experiments details
    failed_results = [r for r in results if r.status == "FAIL"]
    if failed_results:
        lines.append("")
        lines.append("## Failed Experiments Details")
        lines.append("")
        
        for result in failed_results:
            lines.append(f"### {result.experiment_id}")
            lines.append("")
            if result.issues:
                lines.append("**Issues:**")
                for issue in result.issues:
                    lines.append(f"- {issue}")
                lines.append("")
    
    return "\n".join(lines)
