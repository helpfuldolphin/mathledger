#!/usr/bin/env python3
"""
PHASE II — NOT RUN IN PHASE I

U2 Uplift Experiment Auditor
=============================

Audits U2 experiment manifests and logs for cryptographic integrity.
Uses the ManifestVerifier for all cryptographic binding checks.

This module performs ZERO uplift interpretation. It only validates
the integrity of experiment artifacts without drawing any conclusions
about RFL effectiveness.

Usage:
    uv run python experiments/audit_uplift_u2.py --manifest PATH [--prereg PATH] [--config PATH]
    uv run python experiments/audit_uplift_u2.py --scan-dir DIR

Absolute Safeguards:
- Do NOT interpret Phase I logs as uplift evidence.
- All Phase II artifacts must be clearly labeled.
- RFL uses verifiable feedback only.
- Zero uplift interpretation in this module.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path for both module and script execution
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from experiments.manifest_verifier import (
    PHASE_LABEL,
    ManifestVerifier,
    compute_sha256_file,
    load_json_file,
    load_jsonl_file,
    verify_manifest,
)


def audit_manifest(
    manifest_path: Path,
    prereg_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    logs_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Audit a single U2 manifest for cryptographic integrity.
    
    This function performs ZERO uplift interpretation.
    
    Args:
        manifest_path: Path to manifest JSON
        prereg_path: Path to preregistration YAML
        config_path: Path to curriculum config YAML
        logs_path: Path to experiment logs JSONL
    
    Returns:
        Audit result dictionary
    """
    print(f"\n{'='*60}")
    print(f"Auditing: {manifest_path}")
    print(f"{'='*60}")
    
    verifier = ManifestVerifier(
        manifest_path=manifest_path,
        prereg_path=prereg_path,
        config_path=config_path,
    )
    
    verdict = verifier.verify_all(logs_path=logs_path)
    json_report = verifier.generate_json_report()
    
    # Add additional audit-specific checks
    audit_result = {
        "manifest_path": str(manifest_path),
        "verification": json_report,
        "additional_checks": [],
        "overall_verdict": verdict,
    }
    
    # Check for Phase II label in manifest
    if verifier.manifest:
        label = verifier.manifest.get("label", "")
        if "PHASE II" in label:
            audit_result["additional_checks"].append({
                "check": "phase_label",
                "status": "PASS",
                "message": f"Manifest has correct Phase II label: {label}",
            })
        else:
            audit_result["additional_checks"].append({
                "check": "phase_label",
                "status": "FAIL",
                "message": f"Manifest missing Phase II label. Found: {label or 'None'}",
            })
            audit_result["overall_verdict"] = "FAIL"
    
    # Check manifest structure
    if verifier.manifest:
        required_fields = ["slice", "mode", "cycles"]
        missing = [f for f in required_fields if f not in verifier.manifest]
        if missing:
            audit_result["additional_checks"].append({
                "check": "required_fields",
                "status": "FAIL",
                "message": f"Missing required fields: {missing}",
            })
            audit_result["overall_verdict"] = "FAIL"
        else:
            audit_result["additional_checks"].append({
                "check": "required_fields",
                "status": "PASS",
                "message": "All required fields present",
            })
    
    return audit_result


def audit_directory(
    scan_dir: Path,
    prereg_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Scan a directory for U2 manifests and audit each one.
    
    Args:
        scan_dir: Directory to scan for manifest files
        prereg_path: Path to preregistration YAML
        config_path: Path to curriculum config YAML
    
    Returns:
        Directory audit summary
    """
    print(f"\n{'='*60}")
    print(f"Scanning directory: {scan_dir}")
    print(f"{'='*60}")
    
    if not scan_dir.exists():
        print(f"ERROR: Directory not found: {scan_dir}")
        return {
            "scan_dir": str(scan_dir),
            "error": "Directory not found",
            "manifests_found": 0,
            "results": [],
        }
    
    # Find manifest files
    manifest_patterns = ["*manifest*.json", "*_manifest_*.json"]
    manifest_files = []
    for pattern in manifest_patterns:
        manifest_files.extend(scan_dir.glob(pattern))
        manifest_files.extend(scan_dir.glob(f"**/{pattern}"))
    
    # Deduplicate
    manifest_files = list(set(manifest_files))
    
    print(f"Found {len(manifest_files)} manifest files")
    
    results = []
    for manifest_path in sorted(manifest_files):
        # Try to find corresponding log file
        logs_path = None
        if manifest_path.name.startswith("uplift_u2_manifest_"):
            # Derive log path from manifest name
            log_name = manifest_path.name.replace("_manifest_", "_").replace(".json", ".jsonl")
            potential_log = manifest_path.parent / log_name
            if potential_log.exists():
                logs_path = potential_log
        
        result = audit_manifest(
            manifest_path=manifest_path,
            prereg_path=prereg_path,
            config_path=config_path,
            logs_path=logs_path,
        )
        results.append(result)
    
    # Summary
    passed = sum(1 for r in results if r["overall_verdict"] == "PASS")
    failed = sum(1 for r in results if r["overall_verdict"] == "FAIL")
    
    return {
        "label": PHASE_LABEL,
        "scan_dir": str(scan_dir),
        "manifests_found": len(manifest_files),
        "passed": passed,
        "failed": failed,
        "results": results,
    }


def generate_audit_report(
    audit_results: Dict[str, Any],
    output_json: Optional[Path] = None,
    output_md: Optional[Path] = None,
) -> None:
    """
    Generate audit reports in JSON and Markdown formats.
    
    Args:
        audit_results: Audit results dictionary
        output_json: Path to write JSON report
        output_md: Path to write Markdown report
    """
    # Add timestamp
    audit_results["timestamp_utc"] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    audit_results["label"] = PHASE_LABEL
    
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(audit_results, f, indent=2)
        print(f"\nJSON report written to: {output_json}")
    
    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        md_content = generate_markdown_audit_report(audit_results)
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"Markdown report written to: {output_md}")


def generate_markdown_audit_report(audit_results: Dict[str, Any]) -> str:
    """Generate Markdown format audit report."""
    lines = [
        "# U2 Uplift Experiment Audit Report",
        "",
        f"**{PHASE_LABEL}**",
        "",
        f"Generated: {audit_results.get('timestamp_utc', 'N/A')}",
        "",
    ]
    
    # Handle single manifest or directory scan
    if "results" in audit_results:
        # Directory scan
        lines.extend([
            "## Summary",
            "",
            f"- Directory: `{audit_results.get('scan_dir', 'N/A')}`",
            f"- Manifests found: {audit_results.get('manifests_found', 0)}",
            f"- Passed: {audit_results.get('passed', 0)}",
            f"- Failed: {audit_results.get('failed', 0)}",
            "",
            "## Results",
            "",
        ])
        
        for result in audit_results.get("results", []):
            verdict_icon = "✅" if result["overall_verdict"] == "PASS" else "❌"
            lines.append(f"### {verdict_icon} {result['manifest_path']}")
            lines.append("")
            lines.append(f"**Verdict:** {result['overall_verdict']}")
            lines.append("")
            
            # Verification findings
            verification = result.get("verification", {})
            for finding in verification.get("findings", []):
                status_icon = {
                    "PASS": "✅",
                    "FAIL": "❌",
                    "SKIP": "⏭️",
                    "ERROR": "🔴",
                }.get(finding["status"], "❓")
                lines.append(f"- {status_icon} {finding['check']}: {finding['message']}")
            
            # Additional checks
            for check in result.get("additional_checks", []):
                status_icon = "✅" if check["status"] == "PASS" else "❌"
                lines.append(f"- {status_icon} {check['check']}: {check['message']}")
            
            lines.append("")
    else:
        # Single manifest
        verdict_icon = "✅" if audit_results.get("overall_verdict") == "PASS" else "❌"
        lines.extend([
            f"## {verdict_icon} Audit Result",
            "",
            f"**Manifest:** `{audit_results.get('manifest_path', 'N/A')}`",
            f"**Verdict:** {audit_results.get('overall_verdict', 'N/A')}",
            "",
        ])
    
    lines.extend([
        "---",
        "",
        f"*{PHASE_LABEL}*",
        "",
        "*This audit performs ZERO uplift interpretation. It only validates cryptographic integrity.*",
    ])
    
    return "\n".join(lines)


def main():
    """CLI entry point."""
    print(f"{'='*60}")
    print(PHASE_LABEL)
    print("U2 Uplift Experiment Auditor")
    print(f"{'='*60}")
    print()
    print("NOTICE: This tool performs ZERO uplift interpretation.")
    print("It only validates cryptographic integrity of experiment artifacts.")
    print()
    
    parser = argparse.ArgumentParser(
        description=f"Audit U2 experiment manifests. {PHASE_LABEL}",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Absolute Safeguards:
- Do NOT interpret Phase I logs as uplift evidence.
- All Phase II artifacts must be clearly labeled.
- This tool performs ZERO uplift interpretation.
        """,
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--manifest",
        type=str,
        help="Path to single manifest JSON file to audit",
    )
    group.add_argument(
        "--scan-dir",
        type=str,
        help="Directory to scan for manifest files",
    )
    
    parser.add_argument(
        "--prereg",
        type=str,
        default=None,
        help="Path to preregistration YAML file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to curriculum config YAML file",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default=None,
        help="Path to experiment logs JSONL file (for single manifest audit)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write JSON audit report",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Path to write Markdown audit report",
    )
    
    args = parser.parse_args()
    
    # Set defaults
    project_root = Path(__file__).resolve().parents[1]
    prereg_path = Path(args.prereg) if args.prereg else project_root / "experiments" / "prereg" / "PREREG_UPLIFT_U2.yaml"
    config_path = Path(args.config) if args.config else project_root / "config" / "curriculum_uplift_phase2.yaml"
    
    output_json = Path(args.output_json) if args.output_json else None
    output_md = Path(args.output_md) if args.output_md else None
    
    if args.manifest:
        # Single manifest audit
        manifest_path = Path(args.manifest)
        logs_path = Path(args.logs) if args.logs else None
        
        audit_results = audit_manifest(
            manifest_path=manifest_path,
            prereg_path=prereg_path,
            config_path=config_path,
            logs_path=logs_path,
        )
    else:
        # Directory scan
        scan_dir = Path(args.scan_dir)
        audit_results = audit_directory(
            scan_dir=scan_dir,
            prereg_path=prereg_path,
            config_path=config_path,
        )
    
    # Generate reports
    generate_audit_report(
        audit_results=audit_results,
        output_json=output_json,
        output_md=output_md,
    )
    
    # Print summary
    print()
    print("=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    
    if "results" in audit_results:
        # Directory scan
        print(f"Manifests audited: {audit_results.get('manifests_found', 0)}")
        print(f"Passed: {audit_results.get('passed', 0)}")
        print(f"Failed: {audit_results.get('failed', 0)}")
    else:
        # Single manifest
        print(f"Manifest: {audit_results.get('manifest_path', 'N/A')}")
        print(f"Verdict: {audit_results.get('overall_verdict', 'N/A')}")
    
    print()
    print(PHASE_LABEL)
    print("Zero uplift interpretation performed.")
    
    # Exit code
    if "results" in audit_results:
        failed = audit_results.get("failed", 0)
        sys.exit(0 if failed == 0 else 1)
    else:
        verdict = audit_results.get("overall_verdict", "FAIL")
        sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
