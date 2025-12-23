"""
Single Experiment Auditor for U2 Uplift Experiments
====================================================

Audits a single experiment directory for:
- Manifest integrity (JSON validity, hash consistency)
- Artifact existence and completeness
- Result log validation (non-empty JSONL)
- Preregistration hash verification

Outputs JSON and Markdown audit reports.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from attestation.manifest_verifier import (
    compute_sha256_file,
    compute_sha256_json,
    load_and_verify_json,
)


@dataclass
class ArtifactCheck:
    """Result of checking a single artifact."""
    path: str
    exists: bool
    size_bytes: int
    line_count: Optional[int]
    expected_hash: Optional[str]
    actual_hash: Optional[str]
    hash_match: Optional[bool]
    issues: List[str]


@dataclass
class AuditResult:
    """Complete audit result for a single experiment."""
    experiment_id: str
    status: str  # "PASS", "FAIL", "SKIP"
    manifest_path: str
    manifest_valid: bool
    manifest_hash: Optional[str]
    artifacts_checked: List[ArtifactCheck]
    issues: List[str]
    summary: Dict[str, Any]


def audit_experiment(
    experiment_dir: Path,
    repo_root: Optional[Path] = None,
    prereg_hash: Optional[str] = None
) -> AuditResult:
    """
    Audit a single U2 uplift experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        repo_root: Repository root (defaults to experiment_dir.parent.parent,
                   assuming structure: repo_root/experiments/EXP_ID/)
        prereg_hash: Optional preregistration hash to verify against manifest
        
    Returns:
        AuditResult with complete audit findings
    """
    if repo_root is None:
        repo_root = experiment_dir.parent.parent
    
    experiment_id = experiment_dir.name
    manifest_path = experiment_dir / "manifest.json"
    
    issues = []
    artifacts_checked = []
    
    # Load and verify manifest
    manifest_result = load_and_verify_json(manifest_path)
    manifest_valid = manifest_result["valid"]
    manifest_hash = manifest_result["sha256"]
    manifest_data = manifest_result["data"]
    
    if not manifest_valid:
        issues.append(f"Manifest invalid: {manifest_result['error']}")
        return AuditResult(
            experiment_id=experiment_id,
            status="FAIL",
            manifest_path=str(manifest_path),
            manifest_valid=False,
            manifest_hash=None,
            artifacts_checked=[],
            issues=issues,
            summary={"error": "Invalid manifest"}
        )
    
    # Check prereg hash if provided
    if prereg_hash and manifest_data:
        manifest_prereg = manifest_data.get("preregistration_hash")
        if manifest_prereg != prereg_hash:
            issues.append(
                f"Preregistration hash mismatch: "
                f"manifest={manifest_prereg}, expected={prereg_hash}"
            )
    
    # Check artifacts
    artifacts = manifest_data.get("artifacts", {})
    
    # Check logs
    for log_entry in artifacts.get("logs", []):
        artifact_check = _check_log_artifact(log_entry, repo_root)
        artifacts_checked.append(artifact_check)
        issues.extend(artifact_check.issues)
    
    # Check figures
    for fig_entry in artifacts.get("figures", []):
        artifact_check = _check_figure_artifact(fig_entry, repo_root)
        artifacts_checked.append(artifact_check)
        issues.extend(artifact_check.issues)
    
    # Determine overall status
    status = "FAIL" if issues else "PASS"
    
    # Build summary
    summary = {
        "total_artifacts": len(artifacts_checked),
        "total_issues": len(issues),
        "artifacts_with_issues": sum(1 for a in artifacts_checked if a.issues),
    }
    
    return AuditResult(
        experiment_id=experiment_id,
        status=status,
        manifest_path=str(manifest_path),
        manifest_valid=manifest_valid,
        manifest_hash=manifest_hash,
        artifacts_checked=artifacts_checked,
        issues=issues,
        summary=summary
    )


def _check_log_artifact(log_entry: Dict[str, Any], repo_root: Path) -> ArtifactCheck:
    """Check a log artifact entry."""
    path = log_entry["path"]
    filepath = repo_root / path
    issues = []
    
    exists = filepath.exists()
    size_bytes = filepath.stat().st_size if exists else 0
    line_count = None
    expected_hash = log_entry.get("sha256")
    actual_hash = compute_sha256_file(filepath) if exists else None
    hash_match = None
    
    if not exists:
        issues.append(f"Log file missing: {path}")
    else:
        if size_bytes == 0:
            issues.append(f"Log file is empty (0 bytes): {path}")
        
        # Count lines if JSONL
        if log_entry.get("type") == "jsonl":
            try:
                with open(filepath, 'r') as f:
                    line_count = sum(1 for _ in f)
                if line_count == 0:
                    issues.append(f"JSONL file has 0 lines: {path}")
            except Exception as e:
                issues.append(f"Failed to read JSONL: {path} - {str(e)}")
        
        # Check hash
        if expected_hash and actual_hash:
            hash_match = (expected_hash == actual_hash)
            if not hash_match:
                issues.append(
                    f"SHA256 mismatch for {path}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
    
    return ArtifactCheck(
        path=path,
        exists=exists,
        size_bytes=size_bytes,
        line_count=line_count,
        expected_hash=expected_hash,
        actual_hash=actual_hash,
        hash_match=hash_match,
        issues=issues
    )


def _check_figure_artifact(fig_entry: Dict[str, Any], repo_root: Path) -> ArtifactCheck:
    """Check a figure artifact entry."""
    path = fig_entry["path"]
    filepath = repo_root / path
    issues = []
    
    exists = filepath.exists()
    size_bytes = filepath.stat().st_size if exists else 0
    expected_hash = fig_entry.get("sha256")
    actual_hash = compute_sha256_file(filepath) if exists else None
    hash_match = None
    
    if not exists:
        issues.append(f"Figure file missing: {path}")
    else:
        if size_bytes == 0:
            issues.append(f"Figure file is empty (0 bytes): {path}")
        
        # Check hash
        if expected_hash and actual_hash:
            hash_match = (expected_hash == actual_hash)
            if not hash_match:
                issues.append(
                    f"SHA256 mismatch for {path}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
    
    return ArtifactCheck(
        path=path,
        exists=exists,
        size_bytes=size_bytes,
        line_count=None,
        expected_hash=expected_hash,
        actual_hash=actual_hash,
        hash_match=hash_match,
        issues=issues
    )


def render_audit_json(result: AuditResult) -> str:
    """Render audit result as JSON string."""
    data = {
        "experiment_id": result.experiment_id,
        "status": result.status,
        "manifest_path": result.manifest_path,
        "manifest_valid": result.manifest_valid,
        "manifest_hash": result.manifest_hash,
        "issues": result.issues,
        "summary": result.summary,
        "artifacts": [
            {
                "path": a.path,
                "exists": a.exists,
                "size_bytes": a.size_bytes,
                "line_count": a.line_count,
                "expected_hash": a.expected_hash,
                "actual_hash": a.actual_hash,
                "hash_match": a.hash_match,
                "issues": a.issues
            }
            for a in result.artifacts_checked
        ]
    }
    return json.dumps(data, indent=2)


def render_audit_markdown(result: AuditResult) -> str:
    """Render audit result as Markdown report."""
    lines = [
        f"# Audit Report: {result.experiment_id}",
        "",
        f"**Status:** `{result.status}`",
        "",
        f"**Manifest:** `{result.manifest_path}`",
        f"- Valid: {result.manifest_valid}",
        f"- SHA-256: `{result.manifest_hash or 'N/A'}`",
        "",
    ]
    
    if result.issues:
        lines.append("## Issues")
        lines.append("")
        for issue in result.issues:
            lines.append(f"- {issue}")
        lines.append("")
    
    lines.append("## Artifacts")
    lines.append("")
    lines.append("| Path | Exists | Size | Lines | Hash Match | Issues |")
    lines.append("|------|--------|------|-------|------------|--------|")
    
    for artifact in result.artifacts_checked:
        exists_str = "✓" if artifact.exists else "✗"
        size_str = str(artifact.size_bytes) if artifact.exists else "-"
        lines_str = str(artifact.line_count) if artifact.line_count is not None else "-"
        hash_str = "✓" if artifact.hash_match else ("✗" if artifact.hash_match is False else "-")
        issues_str = str(len(artifact.issues)) if artifact.issues else "-"
        
        lines.append(
            f"| `{artifact.path}` | {exists_str} | {size_str} | {lines_str} | {hash_str} | {issues_str} |"
        )
    
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key, value in result.summary.items():
        lines.append(f"- {key}: {value}")
    
    return "\n".join(lines)
