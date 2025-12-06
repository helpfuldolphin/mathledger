#!/usr/bin/env python3
"""
FO & RFL Logs Audit (Cursor L - Sober Truth Mode)

Validates existing log files on disk. Does NOT generate new data.
Only reports what actually exists and its quality.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

# Files to audit
LOG_FILES = [
    "results/fo_baseline.jsonl",
    "results/fo_rfl_50.jsonl",
    "results/fo_rfl_1000.jsonl",
    "results/fo_rfl.jsonl",
]

DYNO_CHART = Path("artifacts/figures/rfl_dyno_chart.png")
ABSTENTION_RATE_CHART = Path("artifacts/figures/rfl_abstention_rate.png")


def load_jsonl(filepath: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load JSONL and return entries + any parse errors."""
    entries = []
    errors = []
    
    if not filepath.exists():
        return entries, [f"File does not exist: {filepath}"]
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: {e}")
    
    return entries, errors


def audit_log_file(filepath: Path) -> Dict[str, Any]:
    """Comprehensive audit of a single log file."""
    result = {
        "file": str(filepath),
        "exists": filepath.exists(),
        "size_bytes": filepath.stat().st_size if filepath.exists() else 0,
        "line_count": 0,
        "valid_entries": 0,
        "parse_errors": [],
        "cycle_analysis": {},
        "schema_analysis": {},
        "abstention_analysis": {},
        "status": "unknown",
        "issues": [],
        "warnings": [],
    }
    
    if not filepath.exists():
        result["status"] = "missing"
        return result
    
    if result["size_bytes"] == 0:
        result["status"] = "empty"
        result["issues"].append("File is 0 bytes")
        return result
    
    entries, parse_errors = load_jsonl(filepath)
    result["parse_errors"] = parse_errors
    result["valid_entries"] = len(entries)
    
    # Count lines (including empty/invalid)
    with open(filepath, 'r') as f:
        result["line_count"] = sum(1 for _ in f)
    
    if not entries:
        result["status"] = "no_valid_entries"
        result["issues"].append("No valid JSON entries found")
        return result
    
    # Cycle analysis
    cycles = [e.get("cycle") for e in entries if "cycle" in e]
    if cycles:
        cycles_set = set(cycles)
        expected_cycles = set(range(len(entries)))
        missing = sorted(list(expected_cycles - cycles_set))
        duplicates = [c for c in cycles if cycles.count(c) > 1]
        
        result["cycle_analysis"] = {
            "min": min(cycles),
            "max": max(cycles),
            "count": len(cycles),
            "unique_count": len(cycles_set),
            "missing": missing[:20],  # First 20 missing
            "missing_count": len(missing),
            "duplicates": list(set(duplicates))[:10],
            "is_contiguous": len(missing) == 0 and len(duplicates) == 0,
        }
        
        if missing:
            result["warnings"].append(f"Missing {len(missing)} cycle indices (first: {missing[:5]})")
        if duplicates:
            result["issues"].append(f"Duplicate cycles found: {list(set(duplicates))[:5]}")
    else:
        result["cycle_analysis"] = {"error": "No 'cycle' field found"}
        result["issues"].append("Missing 'cycle' field in all entries")
    
    # Schema analysis
    sample = entries[0]
    result["schema_analysis"] = {
        "has_cycle": "cycle" in sample,
        "has_status": "status" in sample,
        "has_method": "method" in sample,
        "has_abstention": "abstention" in sample,
        "has_derivation": "derivation" in sample,
        "has_rfl": "rfl" in sample,
        "has_mode": "mode" in sample,
        "top_level_fields": sorted(sample.keys()),
    }
    
    # Check schema version (old vs new)
    has_new_schema = result["schema_analysis"]["has_status"] and result["schema_analysis"]["has_method"] and result["schema_analysis"]["has_abstention"]
    has_old_schema = result["schema_analysis"]["has_derivation"] and "abstained" in sample.get("derivation", {})
    
    if has_new_schema:
        result["schema_analysis"]["version"] = "new"
    elif has_old_schema:
        result["schema_analysis"]["version"] = "old"
        result["warnings"].append("Using old schema (missing status/method/abstention fields)")
    else:
        result["schema_analysis"]["version"] = "unknown"
        result["issues"].append("Schema does not match expected old or new format")
    
    # Abstention analysis
    abstention_values = []
    for entry in entries:
        is_abstention = False
        if "abstention" in entry:
            is_abstention = bool(entry["abstention"])
        elif "status" in entry:
            is_abstention = str(entry["status"]).lower() == "abstain"
        elif "derivation" in entry and isinstance(entry["derivation"], dict):
            is_abstention = entry["derivation"].get("abstained", 0) > 0
        
        abstention_values.append(is_abstention)
    
    if abstention_values:
        abstention_rate = sum(abstention_values) / len(abstention_values)
        result["abstention_analysis"] = {
            "total_abstentions": sum(abstention_values),
            "total_entries": len(abstention_values),
            "abstention_rate": abstention_rate,
            "is_degenerate": abstention_rate == 0.0 or abstention_rate == 1.0,
        }
        
        if abstention_rate == 0.0:
            result["warnings"].append("0% abstention rate (no abstentions found)")
        elif abstention_rate == 1.0:
            result["warnings"].append("100% abstention rate (all entries abstained)")
        elif abstention_rate > 0.95:
            result["warnings"].append(f"Very high abstention rate: {abstention_rate:.1%}")
        elif abstention_rate < 0.05:
            result["warnings"].append(f"Very low abstention rate: {abstention_rate:.1%}")
    
    # Mode analysis
    modes = [e.get("mode") for e in entries if "mode" in e]
    if modes:
        mode_counts = Counter(modes)
        result["schema_analysis"]["mode_distribution"] = dict(mode_counts)
    
    # RFL execution check (for RFL logs)
    if "rfl" in sample:
        rfl_executed = [e.get("rfl", {}).get("executed", False) for e in entries]
        rfl_exec_count = sum(rfl_executed)
        result["schema_analysis"]["rfl_executed_count"] = rfl_exec_count
        result["schema_analysis"]["rfl_executed_rate"] = rfl_exec_count / len(entries) if entries else 0
        
        if "rfl" in str(filepath).lower() and rfl_exec_count == 0:
            result["warnings"].append("RFL log has rfl.executed=false for all entries")
    
    # Determine overall status
    if result["issues"]:
        result["status"] = "has_issues"
    elif result["warnings"]:
        result["status"] = "has_warnings"
    else:
        result["status"] = "ok"
    
    return result


def audit_dyno_chart() -> Dict[str, Any]:
    """Audit the dyno chart file."""
    result = {
        "file": str(DYNO_CHART),
        "exists": DYNO_CHART.exists(),
        "size_bytes": DYNO_CHART.stat().st_size if DYNO_CHART.exists() else 0,
        "is_valid_png": False,
        "issues": [],
        "warnings": [],
    }
    
    if not DYNO_CHART.exists():
        result["issues"].append("Dyno chart file does not exist")
        return result
    
    if result["size_bytes"] == 0:
        result["issues"].append("Dyno chart is 0 bytes (empty)")
        return result
    
    if result["size_bytes"] < 1000:
        result["warnings"].append(f"Dyno chart is very small: {result['size_bytes']} bytes")
    
    # Check PNG header
    try:
        with open(DYNO_CHART, 'rb') as f:
            header = f.read(8)
            if header == b'\x89PNG\r\n\x1a\n':
                result["is_valid_png"] = True
            else:
                result["issues"].append("File does not have valid PNG header")
    except Exception as e:
        result["issues"].append(f"Could not read file: {e}")
    
    # Check if abstention_rate chart also exists (might be the same)
    if ABSTENTION_RATE_CHART.exists():
        result["warnings"].append(f"Alternative chart exists: {ABSTENTION_RATE_CHART} (may be same as dyno chart)")
    
    return result


def find_dyno_chart_data_source() -> Optional[Tuple[str, str]]:
    """Try to determine which log files were used to generate the dyno chart."""
    # Check if there's a manifest or script that generated it
    # For now, we'll infer from file timestamps and sizes
    
    baseline_candidates = [
        "results/fo_baseline.jsonl",
    ]
    
    rfl_candidates = [
        "results/fo_rfl_50.jsonl",
        "results/fo_rfl_1000.jsonl",
        "results/fo_rfl.jsonl",
    ]
    
    # Simple heuristic: if dyno chart exists, check which RFL log is most likely
    # This is imperfect but better than nothing
    if DYNO_CHART.exists():
        baseline_path = None
        rfl_path = None
        
        # Find largest/most complete baseline
        for path_str in baseline_candidates:
            path = Path(path_str)
            if path.exists() and path.stat().st_size > 0:
                baseline_path = path_str
                break
        
        # Find most complete RFL log
        best_rfl = None
        best_size = 0
        for path_str in rfl_candidates:
            path = Path(path_str)
            if path.exists():
                size = path.stat().st_size
                if size > best_size:
                    best_size = size
                    best_rfl = path_str
        
        if baseline_path and best_rfl:
            return (baseline_path, best_rfl)
    
    return None


def compute_numeric_summary(log_results: List[Dict[str, Any]]) -> str:
    """Generate concise numeric summary table."""
    lines = []
    lines.append("=" * 80)
    lines.append("PHASE I NUMERIC SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("| File | Cycles | Abstention Count | Abstention % | Schema |")
    lines.append("|------|--------|------------------|--------------|--------|")
    
    for result in log_results:
        file = Path(result['file']).name
        cycles = result['valid_entries']
        abstention_count = result['abstention_analysis'].get('total_abstentions', 0)
        abstention_pct = result['abstention_analysis'].get('abstention_rate', 0) * 100
        schema = result['schema_analysis'].get('version', 'unknown')
        
        lines.append(f"| {file} | {cycles} | {abstention_count} | {abstention_pct:.1f}% | {schema} |")
    
    lines.append("")
    return "\n".join(lines)


def main():
    """Run comprehensive audit."""
    print("=" * 80)
    print("FO & RFL Logs Audit (Cursor L - Sober Truth Mode)")
    print("=" * 80)
    print()
    
    # Audit all log files
    log_results = []
    for log_file in LOG_FILES:
        path = Path(log_file)
        print(f"Auditing: {log_file}")
        result = audit_log_file(path)
        log_results.append(result)
        print()
    
    # Audit dyno chart
    print("Auditing: Dyno Chart")
    dyno_result = audit_dyno_chart()
    print()
    
    # Print detailed results
    print("=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    print()
    
    for result in log_results:
        print(f"File: {result['file']}")
        print(f"  Status: {result['status']}")
        print(f"  Size: {result['size_bytes']:,} bytes")
        print(f"  Valid entries: {result['valid_entries']}")
        
        if result['cycle_analysis']:
            ca = result['cycle_analysis']
            if 'min' in ca:
                print(f"  Cycles: {ca['min']} to {ca['max']} ({ca['count']} total, {ca['unique_count']} unique)")
                if not ca['is_contiguous']:
                    print(f"    ⚠️  Not contiguous: {ca['missing_count']} missing, {len(ca.get('duplicates', []))} duplicates")
        
        if result['schema_analysis']:
            sa = result['schema_analysis']
            print(f"  Schema: {sa.get('version', 'unknown')}")
            print(f"    Has status: {sa.get('has_status', False)}")
            print(f"    Has method: {sa.get('has_method', False)}")
            print(f"    Has abstention: {sa.get('has_abstention', False)}")
            if 'mode_distribution' in sa:
                print(f"    Modes: {sa['mode_distribution']}")
            if 'rfl_executed_count' in sa:
                print(f"    RFL executed: {sa['rfl_executed_count']}/{result['valid_entries']} ({sa.get('rfl_executed_rate', 0):.1%})")
        
        if result['abstention_analysis']:
            aa = result['abstention_analysis']
            print(f"  Abstention: {aa['abstention_rate']:.1%} ({aa['total_abstentions']}/{aa['total_entries']})")
            if aa['is_degenerate']:
                print(f"    ⚠️  DEGENERATE: {aa['abstention_rate']:.1%} abstention rate")
        
        if result['issues']:
            print(f"  Issues:")
            for issue in result['issues']:
                print(f"    ✗ {issue}")
        
        if result['warnings']:
            print(f"  Warnings:")
            for warning in result['warnings']:
                print(f"    ⚠ {warning}")
        
        print()
    
    # Dyno chart results
    print(f"Dyno Chart: {dyno_result['file']}")
    print(f"  Exists: {dyno_result['exists']}")
    if dyno_result['exists']:
        print(f"  Size: {dyno_result['size_bytes']:,} bytes")
        print(f"  Valid PNG: {dyno_result['is_valid_png']}")
    if dyno_result['issues']:
        print(f"  Issues:")
        for issue in dyno_result['issues']:
            print(f"    ✗ {issue}")
    if dyno_result['warnings']:
        print(f"  Warnings:")
        for warning in dyno_result['warnings']:
            print(f"    ⚠ {warning}")
    print()
    
    # Try to infer data source
    data_source = find_dyno_chart_data_source()
    if data_source:
        baseline_path, rfl_path = data_source
        print(f"Inferred Dyno Chart Data Source:")
        print(f"  Baseline: {baseline_path}")
        print(f"  RFL: {rfl_path}")
        print()
    
    # Numeric Summary
    print(compute_numeric_summary(log_results))
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_files = len(log_results)
    existing_files = sum(1 for r in log_results if r['exists'])
    valid_files = sum(1 for r in log_results if r['status'] == 'ok')
    files_with_issues = sum(1 for r in log_results if r['status'] == 'has_issues')
    
    print(f"Log files: {existing_files}/{total_files} exist, {valid_files} valid, {files_with_issues} have issues")
    print(f"Dyno chart: {'EXISTS' if dyno_result['exists'] else 'MISSING'}")
    
    # Check for 100% abstention in all RFL logs
    rfl_results = [r for r in log_results if 'rfl' in r['file'].lower() and r['exists']]
    all_rfl_100pct = all(
        r['abstention_analysis'].get('abstention_rate', 0) == 1.0 
        for r in rfl_results 
        if r['abstention_analysis']
    )
    
    if all_rfl_100pct and rfl_results:
        print()
        print("⚠️  CRITICAL FINDING: All RFL logs show 100% abstention rate")
        print("   As of Phase I, all RFL logs (including fo_rfl.jsonl) show 100% abstention;")
        print("   there is no empirical uplift yet.")
    
    # Recommendations
    print()
    print("RECOMMENDATIONS:")
    
    # Find best baseline + RFL pair for dyno chart
    baseline_candidates = [r for r in log_results if 'baseline' in r['file'].lower() and r['valid_entries'] > 0]
    rfl_candidates = [r for r in log_results if 'rfl' in r['file'].lower() and r['valid_entries'] > 0]
    
    if baseline_candidates and rfl_candidates:
        best_baseline = max(baseline_candidates, key=lambda x: x['valid_entries'])
        best_rfl = max(rfl_candidates, key=lambda x: x['valid_entries'])
        
        print(f"  ✓ Best baseline log: {best_baseline['file']} ({best_baseline['valid_entries']} entries)")
        print(f"  ✓ Best RFL log: {best_rfl['file']} ({best_rfl['valid_entries']} entries)")
        
        if dyno_result['exists']:
            print(f"  ✓ Dyno chart exists - verify it was generated from these logs")
            print(f"    Document: 'Phase I RFL uplift (N={best_baseline['valid_entries']} baseline, N={best_rfl['valid_entries']} RFL)'")
        else:
            print(f"  ⚠ Generate dyno chart from: {best_baseline['file']} + {best_rfl['file']}")
    
    # Schema warnings
    old_schema_files = [r for r in log_results if r['schema_analysis'].get('version') == 'old']
    if old_schema_files:
        print(f"  ⚠ {len(old_schema_files)} file(s) use old schema (missing status/method/abstention)")
        print(f"    Analysis script can handle this, but new schema is preferred")
    
    # Degenerate abstention
    degenerate_files = [r for r in log_results if r['abstention_analysis'].get('is_degenerate', False)]
    if degenerate_files:
        print(f"  ⚠ {len(degenerate_files)} file(s) have degenerate abstention rates (0% or 100%)")
        print(f"    These may not be suitable for dyno chart comparison")
    
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

