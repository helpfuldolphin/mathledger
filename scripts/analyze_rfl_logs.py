#!/usr/bin/env python3
"""
Analyze RFL log files for Evidence Pack v1 classification.
"""

import json
import hashlib
import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def analyze_log_file(filepath: Path) -> dict:
    """Analyze a single RFL log file."""
    if not filepath.exists():
        return {
            "exists": False,
            "status": "missing"
        }
    
    with open(filepath, 'rb') as f:
        content = f.read()
        sha256 = hashlib.sha256(content).hexdigest()
        size_bytes = len(content)
    
    if size_bytes == 0:
        return {
            "exists": True,
            "status": "empty",
            "size_bytes": 0,
            "sha256": sha256
        }
    
    # Parse JSONL
    lines = content.decode('utf-8').strip().split('\n')
    cycles = []
    all_abstain = True
    all_lean_disabled = True
    has_roots = True
    
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            cycle = record.get('cycle')
            if cycle is not None:
                cycles.append(cycle)
            
            if record.get('status') != 'abstain':
                all_abstain = False
            if record.get('method') != 'lean-disabled':
                all_lean_disabled = False
            if 'roots' not in record:
                has_roots = False
        except json.JSONDecodeError:
            continue
    
    cycles_sorted = sorted(cycles)
    is_contiguous = cycles_sorted == list(range(min(cycles_sorted), max(cycles_sorted) + 1)) if cycles_sorted else False
    
    # Classify
    if len(cycles) == 0:
        classification = "empty"
    elif len(cycles) >= 50 and is_contiguous:
        if all_abstain and all_lean_disabled:
            classification = "plumbing_test"  # All abstentions, lean-disabled = metabolism test
        else:
            classification = "partial"  # Has data but not complete
    elif len(cycles) < 50:
        classification = "small_sample"
    else:
        classification = "partial"
    
    return {
        "exists": True,
        "status": classification,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "line_count": len(lines),
        "cycle_count": len(cycles),
        "first_cycle": min(cycles_sorted) if cycles_sorted else None,
        "last_cycle": max(cycles_sorted) if cycles_sorted else None,
        "is_contiguous": is_contiguous,
        "all_abstain": all_abstain,
        "all_lean_disabled": all_lean_disabled,
        "has_roots": has_roots
    }


def main():
    """Analyze all RFL log files."""
    files_to_check = {
        "fo_rfl_50": REPO_ROOT / "results" / "fo_rfl_50.jsonl",
        "fo_rfl": REPO_ROOT / "results" / "fo_rfl.jsonl",
        "fo_rfl_1000": REPO_ROOT / "results" / "fo_rfl_1000.jsonl",
        "fo_1000_rfl_experiment_log": REPO_ROOT / "artifacts" / "phase_ii" / "fo_series_1" / "fo_1000_rfl" / "experiment_log.jsonl",
    }
    
    results = {}
    for name, path in files_to_check.items():
        results[name] = analyze_log_file(path)
        results[name]["path"] = str(path.relative_to(REPO_ROOT))
    
    # Print results
    print("=" * 80)
    print("RFL LOG FILE ANALYSIS")
    print("=" * 80)
    print()
    
    for name, data in results.items():
        print(f"{name}:")
        print(f"  Path: {data['path']}")
        print(f"  Exists: {data.get('exists', False)}")
        if data.get('exists'):
            print(f"  Status: {data.get('status', 'unknown')}")
            print(f"  Size: {data.get('size_bytes', 0)} bytes")
            print(f"  SHA256: {data.get('sha256', 'N/A')}")
            if data.get('cycle_count') is not None:
                print(f"  Cycles: {data.get('cycle_count')} (range: {data.get('first_cycle')}-{data.get('last_cycle')})")
                print(f"  Contiguous: {data.get('is_contiguous', False)}")
                print(f"  All abstain: {data.get('all_abstain', False)}")
                print(f"  All lean-disabled: {data.get('all_lean_disabled', False)}")
        print()
    
    # Save JSON report
    report_path = REPO_ROOT / "artifacts" / "rfl_logs_analysis.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Report saved to: {report_path}")
    
    return results


if __name__ == "__main__":
    main()

