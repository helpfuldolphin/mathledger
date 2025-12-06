#!/usr/bin/env python3
"""
Evidence Pack v1 Audit Script (Cursor A)

Verifies that all manifest.json files point to real, non-empty files,
and that SHA256 hashes match. Generates a consistency report for Reviewer 2.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

REPO_ROOT = Path(__file__).parent.parent


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    if not filepath.exists():
        return None
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def verify_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Verify a single manifest.json file."""
    if not manifest_path.exists():
        return {
            "status": "error",
            "error": "manifest file not found",
            "path": str(manifest_path)
        }
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "error": f"invalid JSON: {e}",
            "path": str(manifest_path)
        }
    
    issues = []
    verified_artifacts = []
    
    # Check logs
    for log_entry in manifest.get("artifacts", {}).get("logs", []):
        log_path = REPO_ROOT / log_entry["path"]
        expected_sha256 = log_entry.get("sha256")
        
        if not log_path.exists():
            issues.append(f"Log file missing: {log_entry['path']}")
            continue
        
        file_size = log_path.stat().st_size
        actual_sha256 = compute_sha256(log_path)
        
        if file_size == 0:
            issues.append(f"Log file is empty (0 bytes): {log_entry['path']}")
        
        if expected_sha256 and actual_sha256 != expected_sha256:
            issues.append(
                f"SHA256 mismatch for {log_entry['path']}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        
        # Count lines if JSONL
        line_count = 0
        if log_entry.get("type") == "jsonl":
            try:
                with open(log_path, 'r') as f:
                    line_count = sum(1 for _ in f)
            except Exception:
                pass
        
        verified_artifacts.append({
            "type": "log",
            "path": log_entry["path"],
            "exists": True,
            "size_bytes": file_size,
            "line_count": line_count,
            "sha256_match": expected_sha256 == actual_sha256 if expected_sha256 else None
        })
    
    # Check figures
    for fig_entry in manifest.get("artifacts", {}).get("figures", []):
        fig_path = REPO_ROOT / fig_entry["path"]
        expected_sha256 = fig_entry.get("sha256")
        
        if not fig_path.exists():
            issues.append(f"Figure file missing: {fig_entry['path']}")
            continue
        
        file_size = fig_path.stat().st_size
        actual_sha256 = compute_sha256(fig_path)
        
        if file_size == 0:
            issues.append(f"Figure file is empty (0 bytes): {fig_entry['path']}")
        
        if expected_sha256 and actual_sha256 != expected_sha256:
            issues.append(
                f"SHA256 mismatch for {fig_entry['path']}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        
        verified_artifacts.append({
            "type": "figure",
            "path": fig_entry["path"],
            "exists": True,
            "size_bytes": file_size,
            "sha256_match": expected_sha256 == actual_sha256 if expected_sha256 else None
        })
    
    return {
        "status": "ok" if not issues else "inconsistent",
        "experiment_id": manifest.get("experiment", {}).get("id"),
        "experiment_type": manifest.get("experiment", {}).get("type"),
        "issues": issues,
        "artifacts": verified_artifacts,
        "path": str(manifest_path)
    }


def verify_attestation(attestation_path: Path) -> Dict[str, Any]:
    """Verify attestation.json file."""
    if not attestation_path.exists():
        return {
            "status": "error",
            "error": "attestation file not found",
            "path": str(attestation_path)
        }
    
    try:
        with open(attestation_path, 'r') as f:
            attestation = json.load(f)
    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "error": f"invalid JSON: {e}",
            "path": str(attestation_path)
        }
    
    required_fields = ["H_t", "R_t", "U_t"]
    missing_fields = [f for f in required_fields if f not in attestation]
    
    return {
        "status": "ok" if not missing_fields else "incomplete",
        "path": str(attestation_path),
        "H_t": attestation.get("H_t"),
        "R_t": attestation.get("R_t"),
        "U_t": attestation.get("U_t"),
        "missing_fields": missing_fields,
        "version": attestation.get("version"),
        "run_id": attestation.get("run_id")
    }


def audit_all_manifests() -> Dict[str, Any]:
    """Audit all manifest files in the repository."""
    results = {
        "manifests": [],
        "attestations": [],
        "summary": {
            "total_manifests": 0,
            "ok": 0,
            "inconsistent": 0,
            "errors": 0
        }
    }
    
    # Find all manifest.json files
    manifest_paths = [
        REPO_ROOT / "artifacts" / "phase_ii" / "fo_series_1" / "fo_1000_baseline" / "manifest.json",
        REPO_ROOT / "artifacts" / "phase_ii" / "fo_series_1" / "fo_1000_rfl" / "manifest.json",
        REPO_ROOT / "artifacts" / "phase_ii" / "fo_series_1" / "fo_1000_baseline" / "run_20251130_import" / "manifest.json",
        REPO_ROOT / "artifacts" / "phase_ii" / "fo_series_1" / "fo_1000_rfl" / "run_20251130_import" / "manifest.json",
    ]
    
    for manifest_path in manifest_paths:
        if manifest_path.exists():
            result = verify_manifest(manifest_path)
            results["manifests"].append(result)
            results["summary"]["total_manifests"] += 1
            if result["status"] == "ok":
                results["summary"]["ok"] += 1
            elif result["status"] == "inconsistent":
                results["summary"]["inconsistent"] += 1
            else:
                results["summary"]["errors"] += 1
    
    # Verify attestation files
    attestation_paths = [
        REPO_ROOT / "artifacts" / "first_organism" / "attestation.json",
        REPO_ROOT / "artifacts" / "phase_ii" / "fo_series_1" / "fo_1000_rfl" / "run_20251130_import" / "proofs" / "attestation.json",
    ]
    
    for att_path in attestation_paths:
        if att_path.exists():
            result = verify_attestation(att_path)
            results["attestations"].append(result)
    
    return results


def main():
    """Main audit function."""
    print("=" * 80)
    print("EVIDENCE PACK V1 AUDIT (Cursor A - Sober Truth Mode)")
    print("=" * 80)
    print()
    
    results = audit_all_manifests()
    
    # Print summary
    print("SUMMARY")
    print("-" * 80)
    print(f"Total manifests checked: {results['summary']['total_manifests']}")
    print(f"  ✓ OK: {results['summary']['ok']}")
    print(f"  ⚠ Inconsistent: {results['summary']['inconsistent']}")
    print(f"  ✗ Errors: {results['summary']['errors']}")
    print()
    
    # Print manifest details
    print("MANIFEST VERIFICATION")
    print("-" * 80)
    for manifest in results["manifests"]:
        print(f"\nExperiment: {manifest.get('experiment_id', 'unknown')}")
        print(f"  Type: {manifest.get('experiment_type', 'unknown')}")
        print(f"  Status: {manifest['status']}")
        print(f"  Path: {manifest['path']}")
        
        if manifest.get("issues"):
            print(f"  Issues ({len(manifest['issues'])}):")
            for issue in manifest["issues"]:
                print(f"    - {issue}")
        
        if manifest.get("artifacts"):
            print(f"  Artifacts ({len(manifest['artifacts'])}):")
            for artifact in manifest["artifacts"]:
                status = "✓" if artifact.get("exists") and artifact.get("size_bytes", 0) > 0 else "✗"
                print(f"    {status} {artifact['type']}: {artifact['path']}")
                if artifact.get("line_count") is not None:
                    print(f"      Lines: {artifact['line_count']}")
                print(f"      Size: {artifact.get('size_bytes', 0)} bytes")
                if artifact.get("sha256_match") is False:
                    print(f"      ⚠ SHA256 mismatch!")
    
    # Print attestation details
    print("\nATTESTATION VERIFICATION")
    print("-" * 80)
    for att in results["attestations"]:
        print(f"\nPath: {att['path']}")
        print(f"  Status: {att['status']}")
        if att.get("H_t"):
            print(f"  H_t: {att['H_t']}")
        if att.get("R_t"):
            print(f"  R_t: {att['R_t']}")
        if att.get("U_t"):
            print(f"  U_t: {att['U_t']}")
        if att.get("missing_fields"):
            print(f"  ⚠ Missing fields: {att['missing_fields']}")
    
    # Save detailed report
    report_path = REPO_ROOT / "artifacts" / "evidence_pack_v1_audit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_path}")
    print()
    
    # Return exit code based on issues
    if results["summary"]["inconsistent"] > 0 or results["summary"]["errors"] > 0:
        print("⚠ WARNING: Inconsistencies found. Review the report above.")
        return 1
    else:
        print("✓ All manifests verified successfully.")
        return 0


if __name__ == "__main__":
    exit(main())

