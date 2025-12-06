#!/usr/bin/env python3
"""
Baseline CI Verification Tool - Dry-run simulator for baseline stability

Parses two docs_delta.json reports from consecutive CI runs and verifies
baseline stability with Proof-or-Abstain discipline.

Pass-Lines:
- [PASS] Baseline Stable Δ=0 (no drift detected)
- [FAIL] Baseline Drift Detected (+X -Y files)
- ABSTAIN: <reason> (with remediation)

Usage:
    python tools/docs/verify_baseline_ci.py \\
        --run1 artifacts/docs/run1_delta.json \\
        --run2 artifacts/docs/run2_delta.json \\
        --baseline1 docs/methods/baseline_run1.json \\
        --baseline2 docs/methods/baseline_run2.json

    python tools/docs/verify_baseline_ci.py \\
        --run1 artifacts/docs/run1_delta.json \\
        --run2 artifacts/docs/run2_delta.json \\
        --auto-detect-baselines
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from backend.repro.determinism import deterministic_run_id


def load_json_file(path: Path) -> Optional[Dict]:
    """Load JSON file with error handling."""
    try:
        with open(path, "r", encoding="ascii") as f:
            content = f.read()
            
        try:
            content.encode("ascii")
        except UnicodeEncodeError:
            print(f"ABSTAIN: File contains non-ASCII characters: {path}")
            print("Remediation: Run ASCII sweeper on file")
            return None
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"ABSTAIN: Invalid JSON in {path}: {e}")
            print("Remediation: Verify file is valid JSON")
            return None
        
        return data
    
    except FileNotFoundError:
        print(f"ABSTAIN: File not found: {path}")
        print("Remediation: Verify file path is correct")
        return None
    except PermissionError:
        print(f"ABSTAIN: Permission denied reading {path}")
        print("Remediation: Check file permissions (chmod 644)")
        return None
    except OSError as e:
        print(f"ABSTAIN: OS error reading {path}: {e}")
        print("Remediation: Check disk space and file system integrity")
        return None


def validate_delta_report(data: Dict, run_name: str) -> bool:
    """Validate delta report structure."""
    if not isinstance(data, dict):
        print(f"ABSTAIN: {run_name} is not a dictionary")
        print("Remediation: Verify file is a valid docs_delta.json report")
        return False
    
    format_version = data.get("format_version")
    if format_version != "1.0":
        print(f"ABSTAIN: {run_name} format version mismatch (expected 1.0, got {format_version})")
        print("Remediation: Regenerate delta report with current version")
        return False
    
    report_type = data.get("report_type")
    if report_type != "docs_delta":
        print(f"ABSTAIN: {run_name} wrong report type (expected docs_delta, got {report_type})")
        print("Remediation: Ensure file is a docs_delta.json report")
        return False
    
    if "checksums" not in data:
        print(f"ABSTAIN: {run_name} missing 'checksums' key")
        print("Remediation: Regenerate delta report")
        return False
    
    if "delta" not in data:
        print(f"ABSTAIN: {run_name} missing 'delta' key")
        print("Remediation: Regenerate delta report")
        return False
    
    return True


def validate_baseline(data: Dict, baseline_name: str) -> bool:
    """Validate baseline file structure."""
    if not isinstance(data, dict):
        print(f"ABSTAIN: {baseline_name} is not a dictionary")
        print("Remediation: Verify file is a valid baseline file")
        return False
    
    format_version = data.get("format_version")
    if format_version != "1.0":
        print(f"ABSTAIN: {baseline_name} format version mismatch (expected 1.0, got {format_version})")
        print("Remediation: Regenerate baseline with current version")
        return False
    
    baseline_type = data.get("baseline_type")
    if baseline_type != "docs_delta_baseline":
        print(f"ABSTAIN: {baseline_name} wrong baseline type (expected docs_delta_baseline, got {baseline_type})")
        print("Remediation: Ensure file is a docs_delta_baseline.json file")
        return False
    
    if "checksums" not in data:
        print(f"ABSTAIN: {baseline_name} missing 'checksums' key")
        print("Remediation: Regenerate baseline")
        return False
    
    if not isinstance(data["checksums"], dict):
        print(f"ABSTAIN: {baseline_name} checksums is not a dictionary")
        print("Remediation: Regenerate baseline")
        return False
    
    return True


def compute_baseline_hash(baseline_data: Dict) -> str:
    """Compute SHA-256 hash of baseline using RFC 8785 canonicalization."""
    canonical = json.dumps(baseline_data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def compare_baselines(baseline1: Dict, baseline2: Dict) -> Tuple[bool, Dict]:
    """
    Compare two baselines and detect drift.
    
    Returns:
        (stable, drift_info) where stable is True if baselines identical,
        drift_info contains added/removed/modified files if drift detected
    """
    checksums1 = baseline1.get("checksums", {})
    checksums2 = baseline2.get("checksums", {})
    
    hash1 = compute_baseline_hash(baseline1)
    hash2 = compute_baseline_hash(baseline2)
    
    if hash1 == hash2:
        return True, {
            "hash1": hash1,
            "hash2": hash2,
            "added": [],
            "removed": [],
            "modified": []
        }
    
    files1 = set(checksums1.keys())
    files2 = set(checksums2.keys())
    
    added = sorted(files2 - files1)
    removed = sorted(files1 - files2)
    modified = sorted([
        f for f in files1 & files2
        if checksums1[f] != checksums2[f]
    ])
    
    return False, {
        "hash1": hash1,
        "hash2": hash2,
        "added": added,
        "removed": removed,
        "modified": modified
    }


def verify_signature(verification_data: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Verify signature of verification JSON output.
    
    Args:
        verification_data: Verification output dictionary with signature field
    
    Returns:
        (valid, expected_sig, found_sig) where valid is True if signature matches
    """
    if "signature" not in verification_data:
        return False, None, None
    
    stored_signature = verification_data["signature"]
    
    data_without_sig = {k: v for k, v in verification_data.items() if k != "signature"}
    
    # Recompute signature
    canonical = json.dumps(data_without_sig, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    computed_signature = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    
    return stored_signature == computed_signature, stored_signature, computed_signature


def generate_drift_html(drift_info: Dict, baseline1_data: Dict, baseline2_data: Dict) -> str:
    """
    Generate HTML drift visualization report.
    
    Args:
        drift_info: Drift information from compare_baselines
        baseline1_data: First baseline data
        baseline2_data: Second baseline data
    
    Returns:
        HTML string with drift visualization
    """
    checksums1 = baseline1_data.get("checksums", {})
    checksums2 = baseline2_data.get("checksums", {})
    
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset=\"ASCII\">",
        "<title>Baseline Drift Report</title>",
        "<style>",
        "body { font-family: monospace; margin: 20px; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid black; padding: 8px; text-align: left; }",
        "th { background-color: lightgray; }",
        ".added { background-color: lightgreen; }",
        ".removed { background-color: lightcoral; }",
        ".modified { background-color: lightyellow; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Baseline Drift Report</h1>",
        f"<p>Baseline 1 SHA-256: <code>{drift_info['hash1']}</code></p>",
        f"<p>Baseline 2 SHA-256: <code>{drift_info['hash2']}</code></p>",
        "<table>",
        "<tr><th>File</th><th>Status</th><th>Hash Before</th><th>Hash After</th></tr>",
    ]
    
    for file in drift_info["added"]:
        hash_after = checksums2.get(file, "")
        html_lines.append(f"<tr class=\"added\"><td>{file}</td><td>Added</td><td>-</td><td>{hash_after}</td></tr>")
    
    for file in drift_info["removed"]:
        hash_before = checksums1.get(file, "")
        html_lines.append(f"<tr class=\"removed\"><td>{file}</td><td>Removed</td><td>{hash_before}</td><td>-</td></tr>")
    
    for file in drift_info["modified"]:
        hash_before = checksums1.get(file, "")
        hash_after = checksums2.get(file, "")
        html_lines.append(f"<tr class=\"modified\"><td>{file}</td><td>Modified</td><td>{hash_before}</td><td>{hash_after}</td></tr>")
    
    html_lines.extend([
        "</table>",
        "</body>",
        "</html>"
    ])
    
    return "\n".join(html_lines)


def generate_drift_jsonl(drift_info: Dict, baseline1_data: Dict, baseline2_data: Dict) -> List[str]:
    """
    Generate JSONL drift report (RFC 8785 canonical JSON per line).
    
    Args:
        drift_info: Drift information from compare_baselines
        baseline1_data: First baseline data
        baseline2_data: Second baseline data
    
    Returns:
        List of JSONL lines (one JSON object per line)
    """
    checksums1 = baseline1_data.get("checksums", {})
    checksums2 = baseline2_data.get("checksums", {})
    
    jsonl_lines = []
    
    header = {
        "record_type": "header",
        "format_version": "1.0",
        "baseline1_sha256": drift_info["hash1"],
        "baseline2_sha256": drift_info["hash2"],
        "drift_summary": {
            "added": len(drift_info["added"]),
            "removed": len(drift_info["removed"]),
            "modified": len(drift_info["modified"])
        }
    }
    jsonl_lines.append(json.dumps(header, sort_keys=True, separators=(',', ':'), ensure_ascii=True))
    
    for file in drift_info["added"]:
        record = {
            "record_type": "file_change",
            "file": file,
            "status": "added",
            "hash_before": None,
            "hash_after": checksums2.get(file)
        }
        jsonl_lines.append(json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=True))
    
    for file in drift_info["removed"]:
        record = {
            "record_type": "file_change",
            "file": file,
            "status": "removed",
            "hash_before": checksums1.get(file),
            "hash_after": None
        }
        jsonl_lines.append(json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=True))
    
    for file in drift_info["modified"]:
        record = {
            "record_type": "file_change",
            "file": file,
            "status": "modified",
            "hash_before": checksums1.get(file),
            "hash_after": checksums2.get(file)
        }
        jsonl_lines.append(json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=True))
    
    return jsonl_lines


def generate_verification_output(stable: bool, drift_info: Dict, error: Optional[str] = None) -> Dict:
    """
    Generate RFC 8785 canonical verification output.
    
    Args:
        stable: Whether baselines are stable
        drift_info: Drift information from compare_baselines
        error: Error message if ABSTAIN
    
    Returns:
        Dictionary with verification results
    """
    if error:
        return {
            "format_version": "1.0",
            "verification_type": "baseline_verification",
            "result": "ABSTAIN",
            "reason": error,
            "remediation": "See error message for remediation steps"
        }
    
    added_count = len(drift_info["added"])
    removed_count = len(drift_info["removed"])
    modified_count = len(drift_info["modified"])
    
    if stable:
        result_data = {
            "format_version": "1.0",
            "verification_type": "baseline_verification",
            "result": "PASS",
            "message": "Baseline Stable Δ=0",
            "baseline1_sha256": drift_info["hash1"],
            "baseline2_sha256": drift_info["hash2"],
            "drift": {
                "added": added_count,
                "removed": removed_count,
                "modified": modified_count
            }
        }
    else:
        result_data = {
            "format_version": "1.0",
            "verification_type": "baseline_verification",
            "result": "FAIL",
            "message": f"Baseline Drift Detected add={added_count} rm={removed_count} mod={modified_count}",
            "baseline1_sha256": drift_info["hash1"],
            "baseline2_sha256": drift_info["hash2"],
            "drift": {
                "added": added_count,
                "removed": removed_count,
                "modified": modified_count
            },
            "files": {
                "added": drift_info["added"],
                "removed": drift_info["removed"],
                "modified": drift_info["modified"]
            },
            "remediation": [
                "If drift is expected (docs were modified), this is normal",
                "If drift is unexpected, investigate which files changed and why",
                "Review git log for documentation changes between runs",
                "Verify baseline persistence is working correctly"
            ]
        }
    
    canonical = json.dumps(result_data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    signature = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    result_data["signature"] = signature
    
    return result_data


def main():
    parser = argparse.ArgumentParser(
        description="Baseline CI Verification - Dry-run simulator for baseline stability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/docs/verify_baseline_ci.py \\
      --run1 artifacts/docs/run1_delta.json \\
      --run2 artifacts/docs/run2_delta.json \\
      --baseline1 docs/methods/baseline_run1.json \\
      --baseline2 docs/methods/baseline_run2.json
  
  python tools/docs/verify_baseline_ci.py \\
      --run1 artifacts/docs/run1_delta.json \\
      --run2 artifacts/docs/run2_delta.json \\
      --auto-detect-baselines
  
  python tools/docs/verify_baseline_ci.py \\
      --baseline1 docs/methods/baseline_previous.json \\
      --baseline2 docs/methods/docs_delta_baseline.json \\
      --baseline-only

Pass-Lines:
  [PASS] Baseline Stable Δ=0 (no drift detected)
  [FAIL] Baseline Drift Detected (+X -Y ~Z files)
  ABSTAIN: <reason> (with remediation)
"""
    )
    
    parser.add_argument(
        "--run1",
        type=Path,
        help="First CI run delta report (docs_delta.json)",
    )
    parser.add_argument(
        "--run2",
        type=Path,
        help="Second CI run delta report (docs_delta.json)",
    )
    parser.add_argument(
        "--baseline1",
        type=Path,
        help="First baseline file (docs_delta_baseline.json)",
    )
    parser.add_argument(
        "--baseline2",
        type=Path,
        help="Second baseline file (docs_delta_baseline.json)",
    )
    parser.add_argument(
        "--auto-detect-baselines",
        action="store_true",
        help="Auto-detect baseline files from delta reports (looks for 'checksums' key)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Compare baselines directly without delta reports",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--json-only",
        type=Path,
        help="Suppress human text; write only RFC 8785 JSON to specified path",
    )
    parser.add_argument(
        "--verify-signature",
        type=Path,
        help="Verify signature of existing verification JSON output",
    )
    parser.add_argument(
        "--emit-drift-report",
        action="store_true",
        help="Generate HTML and JSONL drift visualization reports",
    )
    parser.add_argument(
        "--emit-artifact-metadata",
        type=Path,
        help="Write artifact metadata JSON with run ID and hashes",
    )
    
    args = parser.parse_args()
    
    if args.verify_signature:
        verification_data = load_json_file(args.verify_signature)
        if verification_data is None:
            return 1
        
        valid, expected_sig, found_sig = verify_signature(verification_data)
        
        if valid:
            print(f"[PASS] Baseline Signature verified=true")
            print(f"Signature: {expected_sig}")
            return 0
        else:
            print(f"[FAIL] Baseline Signature mismatch expected={expected_sig} found={found_sig}")
            print("Remediation: Regenerate verification output or check for tampering")
            return 1
    
    if args.baseline_only:
        if not args.baseline1 or not args.baseline2:
            print("ABSTAIN: --baseline-only requires --baseline1 and --baseline2")
            print("Remediation: Provide both baseline file paths")
            return 1
    else:
        if not args.run1 or not args.run2:
            print("ABSTAIN: Requires --run1 and --run2 delta reports")
            print("Remediation: Provide both CI run delta report paths")
            return 1
        
        if not args.auto_detect_baselines and (not args.baseline1 or not args.baseline2):
            print("ABSTAIN: Requires --baseline1 and --baseline2, or use --auto-detect-baselines")
            print("Remediation: Provide baseline file paths or enable auto-detection")
            return 1
    
    if args.baseline_only:
        if not args.json_only:
            print(f"Loading baseline 1 from {args.baseline1}...")
        baseline1_data = load_json_file(args.baseline1)
        if baseline1_data is None:
            return 1
        
        if not args.json_only:
            print(f"Loading baseline 2 from {args.baseline2}...")
        baseline2_data = load_json_file(args.baseline2)
        if baseline2_data is None:
            return 1
        
        if not validate_baseline(baseline1_data, "baseline1"):
            return 1
        if not validate_baseline(baseline2_data, "baseline2"):
            return 1
    
    else:
        if not args.json_only:
            print(f"Loading run 1 delta report from {args.run1}...")
        run1_data = load_json_file(args.run1)
        if run1_data is None:
            return 1
        
        if not args.json_only:
            print(f"Loading run 2 delta report from {args.run2}...")
        run2_data = load_json_file(args.run2)
        if run2_data is None:
            return 1
        
        if not validate_delta_report(run1_data, "run1"):
            return 1
        if not validate_delta_report(run2_data, "run2"):
            return 1
        
        if args.auto_detect_baselines:
            if not args.json_only:
                print("Auto-detecting baselines from delta reports...")
            baseline1_data = {
                "format_version": "1.0",
                "baseline_type": "docs_delta_baseline",
                "checksums": run1_data["checksums"]
            }
            baseline2_data = {
                "format_version": "1.0",
                "baseline_type": "docs_delta_baseline",
                "checksums": run2_data["checksums"]
            }
        else:
            if not args.json_only:
                print(f"Loading baseline 1 from {args.baseline1}...")
            baseline1_data = load_json_file(args.baseline1)
            if baseline1_data is None:
                return 1
            
            if not args.json_only:
                print(f"Loading baseline 2 from {args.baseline2}...")
            baseline2_data = load_json_file(args.baseline2)
            if baseline2_data is None:
                return 1
            
            if not validate_baseline(baseline1_data, "baseline1"):
                return 1
            if not validate_baseline(baseline2_data, "baseline2"):
                return 1
    
    if not args.json_only:
        print("\nComparing baselines...")
    
    stable, drift_info = compare_baselines(baseline1_data, baseline2_data)
    
    added_count = len(drift_info["added"])
    removed_count = len(drift_info["removed"])
    modified_count = len(drift_info["modified"])
    
    verification_output = generate_verification_output(stable, drift_info)
    
    if args.json_only:
        canonical_json = json.dumps(verification_output, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
        args.json_only.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_only, "w", encoding="ascii") as f:
            f.write(canonical_json)
        
        if args.emit_drift_report and not stable:
            drift_count = added_count + removed_count + modified_count
            
            html_output = generate_drift_html(drift_info, baseline1_data, baseline2_data)
            html_path = args.json_only.parent / "baseline_drift_report.html"
            with open(html_path, "w", encoding="ascii") as f:
                f.write(html_output)
            
            jsonl_lines = generate_drift_jsonl(drift_info, baseline1_data, baseline2_data)
            jsonl_path = args.json_only.parent / "baseline_drift_report.jsonl"
            with open(jsonl_path, "w", encoding="ascii") as f:
                f.write("\n".join(jsonl_lines))
            
            print(f"[PASS] Drift Visualization generated files={drift_count}")
        
        if args.emit_artifact_metadata:
            metadata = {
                "format_version": "1.0",
                "artifact_type": "baseline_verification_metadata",
                "run_id": deterministic_run_id("run", drift_info["hash1"], drift_info["hash2"]),
                "baseline1_sha256": drift_info["hash1"],
                "baseline2_sha256": drift_info["hash2"],
                "verification_sha256": verification_output["signature"],
                "stable": stable
            }
            canonical_metadata = json.dumps(metadata, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
            args.emit_artifact_metadata.parent.mkdir(parents=True, exist_ok=True)
            with open(args.emit_artifact_metadata, "w", encoding="ascii") as f:
                f.write(canonical_metadata)
        
        return 0 if stable else 1
    
    if args.verbose:
        print(f"Baseline 1 SHA-256: {drift_info['hash1']}")
        print(f"Baseline 2 SHA-256: {drift_info['hash2']}")
    
    if stable:
        print(f"\n[PASS] Baseline Stable Δ=0")
        print(f"Baseline hashes identical: {drift_info['hash1']}")
        return 0
    else:
        print(f"\n[FAIL] Baseline Drift Detected add={added_count} rm={removed_count} mod={modified_count}")
        print(f"Baseline 1 SHA-256: {drift_info['hash1']}")
        print(f"Baseline 2 SHA-256: {drift_info['hash2']}")
        
        if drift_info["added"]:
            print(f"\nAdded files ({added_count}):")
            for f in drift_info["added"]:
                print(f"  + {f}")
        
        if drift_info["removed"]:
            print(f"\nRemoved files ({removed_count}):")
            for f in drift_info["removed"]:
                print(f"  - {f}")
        
        if drift_info["modified"]:
            print(f"\nModified files ({modified_count}):")
            for f in drift_info["modified"]:
                print(f"  ~ {f}")
        
        print("\nRemediation:")
        print("- If drift is expected (docs were modified), this is normal")
        print("- If drift is unexpected, investigate which files changed and why")
        print("- Review git log for documentation changes between runs")
        print("- Verify baseline persistence is working correctly")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
