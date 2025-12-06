#!/usr/bin/env python3
"""
Docs Delta Watcher for MathLedger V3

Computes checksums over docs assets, compares to last run, and generates
RFC 8785 canonicalized delta report with SHA-256 seal.

Usage:
    python tools/docs/docs_delta.py --docs-dir docs/methods
    python tools/docs/docs_delta.py --docs-dir docs/methods --baseline docs_delta_baseline.json
    uv run python tools/docs/docs_delta.py --rfcsign --out artifacts/docs/docs_delta.json

Fleet Directive: Determinism > speed, RFC 8785, Proof-or-Abstain
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import canonicaljson
    CANONICALJSON_AVAILABLE = True
except ImportError:
    CANONICALJSON_AVAILABLE = False


def canonicalize_json_rfc8785(obj):
    """
    RFC 8785 JSON Canonicalization Scheme (JCS) implementation.
    Ensures deterministic JSON serialization for hash consistency.
    """
    if obj is None:
        return "null"
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    elif isinstance(obj, (int, float)):
        return json.dumps(obj)
    elif isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=True)
    elif isinstance(obj, list):
        items = [canonicalize_json_rfc8785(item) for item in obj]
        return "[" + ",".join(items) + "]"
    elif isinstance(obj, dict):
        sorted_keys = sorted(obj.keys())
        pairs = [
            json.dumps(k, ensure_ascii=True) + ":" + canonicalize_json_rfc8785(obj[k])
            for k in sorted_keys
        ]
        return "{" + ",".join(pairs) + "}"
    else:
        raise TypeError(f"Unsupported type for canonicalization: {type(obj)}")


def validate_rfc8785_against_reference(obj) -> bool:
    """
    Validate custom RFC 8785 implementation against reference library.
    Returns True if implementations match, False otherwise.
    """
    if not CANONICALJSON_AVAILABLE:
        print("WARNING: canonicaljson not available, skipping RFC 8785 validation")
        return True
    
    try:
        custom_output = canonicalize_json_rfc8785(obj)
        reference_output = canonicaljson.encode_canonical_json(obj).decode('ascii')
        
        if custom_output != reference_output:
            print(f"RFC 8785 MISMATCH:")
            print(f"  Custom:    {custom_output[:100]}...")
            print(f"  Reference: {reference_output[:100]}...")
            return False
        
        return True
    except Exception as e:
        print(f"RFC 8785 validation error: {e}")
        return False


def compute_file_checksum(file_path: Path) -> str:
    """Compute SHA-256 checksum of file."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def scan_docs_assets(docs_dir: Path) -> Dict[str, str]:
    """Scan docs directory and compute checksums for all assets."""
    checksums = {}
    
    if not docs_dir.exists():
        print(f"ABSTAIN: Docs directory not found: {docs_dir}")
        return None
    
    for file_path in sorted(docs_dir.rglob("*")):
        if file_path.is_file():
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                    try:
                        content.decode("ascii")
                    except UnicodeDecodeError:
                        print(f"ABSTAIN: Non-ASCII content in {file_path}")
                        return None
                    
                    checksum = hashlib.sha256(content).hexdigest()
                    rel_path = str(file_path.relative_to(docs_dir.parent))
                    checksums[rel_path] = checksum
            except Exception as e:
                print(f"ABSTAIN: Failed to read {file_path}: {e}")
                return None
    
    return checksums


def compute_delta(current: Dict[str, str], baseline: Optional[Dict[str, str]]) -> Dict:
    """Compute delta between current and baseline checksums."""
    if baseline is None:
        return {
            "added": list(current.keys()),
            "removed": [],
            "modified": [],
            "unchanged": []
        }
    
    added = []
    removed = []
    modified = []
    unchanged = []
    
    for path in sorted(set(current.keys()) | set(baseline.keys())):
        if path in current and path not in baseline:
            added.append(path)
        elif path not in current and path in baseline:
            removed.append(path)
        elif current[path] != baseline[path]:
            modified.append(path)
        else:
            unchanged.append(path)
    
    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "unchanged": unchanged
    }


def generate_failure_lens(checksums: Dict[str, str], cross_ledger_index_path: Path) -> Optional[Dict]:
    """Generate failure lens enumerating broken cross-links and missing artifacts."""
    failures = {
        "missing_artifacts": [],
        "broken_cross_links": [],
        "non_ascii_files": []
    }
    
    if not cross_ledger_index_path.exists():
        failures["broken_cross_links"].append({
            "type": "missing_index",
            "path": str(cross_ledger_index_path),
            "description": "Cross-ledger index file not found"
        })
        return failures
    
    try:
        with open(cross_ledger_index_path, "r") as f:
            index_content = f.read()
            index = json.loads(index_content)
    except Exception as e:
        failures["broken_cross_links"].append({
            "type": "invalid_index",
            "path": str(cross_ledger_index_path),
            "description": f"Failed to parse cross-ledger index: {e}"
        })
        return failures
    
    if "sections" not in index:
        failures["broken_cross_links"].append({
            "type": "malformed_index",
            "path": str(cross_ledger_index_path),
            "description": "Cross-ledger index missing 'sections' key"
        })
        return failures
    
    for section, links in index["sections"].items():
        for link in links:
            artifact_path = link.get("artifact_path")
            artifact_hash = link.get("artifact_hash")
            
            if not artifact_path:
                failures["broken_cross_links"].append({
                    "type": "missing_artifact_path",
                    "section": section,
                    "description": "Cross-link missing artifact_path"
                })
                continue
            
            artifact_file = Path(artifact_path)
            if not artifact_file.exists():
                failures["missing_artifacts"].append({
                    "artifact_path": artifact_path,
                    "section": section,
                    "expected_hash": artifact_hash,
                    "description": f"Artifact referenced in section {section} not found"
                })
            else:
                actual_hash = compute_file_checksum(artifact_file)
                if actual_hash != artifact_hash:
                    failures["broken_cross_links"].append({
                        "type": "hash_mismatch",
                        "artifact_path": artifact_path,
                        "section": section,
                        "expected_hash": artifact_hash,
                        "actual_hash": actual_hash,
                        "description": f"Artifact hash mismatch for section {section}"
                    })
    
    return failures


def main():
    parser = argparse.ArgumentParser(
        description="Docs Delta Watcher - Track documentation changes with RFC 8785 canonicalization"
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs/methods"),
        help="Documentation directory to scan",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline docs_delta.json to compare against",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/docs/docs_delta.json"),
        help="Output path for delta report",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Alias for --output (for compatibility with uv run command)",
    )
    parser.add_argument(
        "--failure-lens",
        type=Path,
        default=Path("artifacts/docs/failure_lens.json"),
        help="Output path for failure lens",
    )
    parser.add_argument(
        "--rfcsign",
        action="store_true",
        help="Enable RFC 8785 signature mode (always enabled, flag for compatibility)",
    )
    parser.add_argument(
        "--write-baseline",
        type=Path,
        help="Write baseline file for future comparisons (typically artifacts/docs/docs_delta_baseline.json)",
    )
    args = parser.parse_args()
    
    if args.out:
        args.output = args.out

    print(f"Scanning docs assets in {args.docs_dir}...")
    current_checksums = scan_docs_assets(args.docs_dir)
    
    if current_checksums is None:
        print("ABSTAIN: Failed to scan docs assets")
        return 1
    
    print(f"Scanned {len(current_checksums)} docs assets")
    
    baseline_checksums = None
    if args.baseline and args.baseline.exists():
        print(f"Loading baseline from {args.baseline}...")
        try:
            with open(args.baseline, "r") as f:
                baseline_content = f.read()
                
                try:
                    baseline_content.encode("ascii")
                except UnicodeEncodeError:
                    print("ABSTAIN: Baseline contains non-ASCII characters")
                    print("Remediation: Run ASCII sweeper on baseline file or regenerate baseline")
                    print("Continuing without baseline (all files will show as 'added')")
                    baseline_checksums = None
                else:
                    try:
                        baseline_data = json.loads(baseline_content)
                    except json.JSONDecodeError as e:
                        print(f"ABSTAIN: Baseline is corrupt (invalid JSON): {e}")
                        print("Remediation: Delete baseline file and regenerate with --write-baseline")
                        print("Continuing without baseline (all files will show as 'added')")
                        baseline_checksums = None
                    else:
                        format_version = baseline_data.get("format_version")
                        if format_version != "1.0":
                            print(f"ABSTAIN: Baseline format version mismatch (expected 1.0, got {format_version})")
                            print("Remediation: Regenerate baseline with current version using --write-baseline")
                            print("Continuing without baseline (all files will show as 'added')")
                            baseline_checksums = None
                        else:
                            baseline_type = baseline_data.get("baseline_type")
                            if baseline_type != "docs_delta_baseline":
                                print(f"ABSTAIN: Wrong baseline type (expected docs_delta_baseline, got {baseline_type})")
                                print("Remediation: Ensure you're using the correct baseline file")
                                print("Continuing without baseline (all files will show as 'added')")
                                baseline_checksums = None
                            else:
                                # Extract checksums
                                baseline_checksums = baseline_data.get("checksums")
                                if baseline_checksums is None:
                                    print("ABSTAIN: Baseline missing 'checksums' key")
                                    print("Remediation: Regenerate baseline with --write-baseline")
                                    print("Continuing without baseline (all files will show as 'added')")
                                    baseline_checksums = None
                                elif not isinstance(baseline_checksums, dict):
                                    print(f"ABSTAIN: Baseline checksums is not a dictionary (got {type(baseline_checksums).__name__})")
                                    print("Remediation: Regenerate baseline with --write-baseline")
                                    print("Continuing without baseline (all files will show as 'added')")
                                    baseline_checksums = None
                                else:
                                    print(f"Loaded baseline with {len(baseline_checksums)} checksums")
        except PermissionError:
            print(f"ABSTAIN: Permission denied reading baseline file: {args.baseline}")
            print("Remediation: Check file permissions (chmod 644) or run with appropriate user")
            print("Continuing without baseline (all files will show as 'added')")
            baseline_checksums = None
        except OSError as e:
            print(f"ABSTAIN: OS error reading baseline file: {e}")
            print("Remediation: Check disk space and file system integrity")
            print("Continuing without baseline (all files will show as 'added')")
            baseline_checksums = None
    
    print("Computing delta...")
    delta = compute_delta(current_checksums, baseline_checksums)
    
    print(f"Delta: +{len(delta['added'])} -{len(delta['removed'])} ~{len(delta['modified'])} ={len(delta['unchanged'])}")
    
    if delta["added"]:
        print(f"  Added: {', '.join(delta['added'])}")
    if delta["removed"]:
        print(f"  Removed: {', '.join(delta['removed'])}")
    if delta["modified"]:
        print(f"  Modified: {', '.join(delta['modified'])}")
    
    cross_ledger_index_path = args.docs_dir / "cross_ledger_index.json"
    print(f"\nGenerating failure lens...")
    failures = generate_failure_lens(current_checksums, cross_ledger_index_path)
    
    if failures is None:
        print("ABSTAIN: Failed to generate failure lens")
        return 1
    
    total_failures = (
        len(failures["missing_artifacts"]) +
        len(failures["broken_cross_links"]) +
        len(failures["non_ascii_files"])
    )
    
    if total_failures > 0:
        print(f"FAILURE LENS: {total_failures} issues detected")
        if failures["missing_artifacts"]:
            print(f"  Missing artifacts: {len(failures['missing_artifacts'])}")
            for fail in failures["missing_artifacts"]:
                print(f"    - {fail['artifact_path']} (section {fail['section']})")
        if failures["broken_cross_links"]:
            print(f"  Broken cross-links: {len(failures['broken_cross_links'])}")
            for fail in failures["broken_cross_links"]:
                print(f"    - {fail.get('type', 'unknown')}: {fail.get('description', 'no description')}")
        if failures["non_ascii_files"]:
            print(f"  Non-ASCII files: {len(failures['non_ascii_files'])}")
    else:
        print("FAILURE LENS: No issues detected")
    
    delta_report = {
        "format_version": "1.0",
        "report_type": "docs_delta",
        "checksums": current_checksums,
        "delta": delta,
        "failures": failures
    }
    
    if not validate_rfc8785_against_reference(delta_report):
        print("ABSTAIN: RFC8785 mismatch with reference")
        return 1
    
    canonical_json = canonicalize_json_rfc8785(delta_report)
    
    try:
        canonical_json.encode("ascii")
    except UnicodeEncodeError as e:
        print(f"ABSTAIN: Non-ASCII content in delta report: {e}")
        return 1
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="ascii") as f:
        f.write(canonical_json)
    
    delta_hash = hashlib.sha256(canonical_json.encode("ascii")).hexdigest()
    print(f"\nDelta report written to {args.output}")
    print(f"[PASS] Docs Delta: <sha256:{delta_hash}>")
    
    if args.write_baseline:
        baseline_data = {
            "format_version": "1.0",
            "baseline_type": "docs_delta_baseline",
            "checksums": current_checksums
        }
        baseline_canonical = canonicalize_json_rfc8785(baseline_data)
        args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_baseline, "w", encoding="ascii") as f:
            f.write(baseline_canonical)
        baseline_hash = hashlib.sha256(baseline_canonical.encode("ascii")).hexdigest()
        print(f"Baseline written to {args.write_baseline}")
        print(f"Baseline SHA-256: {baseline_hash}")
    
    if total_failures > 0:
        failure_lens_canonical = canonicalize_json_rfc8785(failures)
        with open(args.failure_lens, "w", encoding="ascii") as f:
            f.write(failure_lens_canonical)
        
        failure_hash = hashlib.sha256(failure_lens_canonical.encode("ascii")).hexdigest()
        print(f"Failure lens written to {args.failure_lens}")
        print(f"Failure lens SHA-256: {failure_hash}")
        
        print("\nABSTAIN: Failures detected - see failure_lens.json")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
