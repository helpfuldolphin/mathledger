#!/usr/bin/env python3
"""
Precise FO Logs Audit (Cursor B)
Computes exact metrics for all Phase I log files.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

def audit_log(filepath: Path) -> Dict[str, Any]:
    """Audit a single log file with precise metrics."""
    result = {
        "file": str(filepath),
        "exists": False,
        "empty": False,
        "line_count": 0,
        "cycle_range": None,
        "cycles_contiguous": False,
        "schema": "UNKNOWN",
        "abstention_stats": {},
        "status_distribution": {},
        "method_distribution": {},
    }
    
    if not filepath.exists():
        return result
    
    result["exists"] = True
    
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num} in {filepath}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return result
    
    if not records:
        result["empty"] = True
        return result
    
    result["line_count"] = len(records)
    
    # Extract cycles
    cycles = []
    for r in records:
        if "cycle" in r:
            cycles.append(int(r["cycle"]))
    
    if cycles:
        result["cycle_range"] = (min(cycles), max(cycles))
        expected_cycles = set(range(min(cycles), max(cycles) + 1))
        actual_cycles = set(cycles)
        result["cycles_contiguous"] = (expected_cycles == actual_cycles and len(cycles) == len(actual_cycles))
    
    # Determine schema
    sample = records[0]
    has_slice_name = "slice_name" in sample
    has_status = "status" in sample
    has_method = "method" in sample
    has_abstention = "abstention" in sample
    
    if has_slice_name and has_status and has_method and has_abstention:
        result["schema"] = "NEW"
    elif not has_slice_name and not has_status and not has_method and not has_abstention:
        result["schema"] = "OLD"
    else:
        result["schema"] = "MIXED"
    
    # Abstention stats
    abstention_count = 0
    abstention_true_count = 0
    abstention_false_count = 0
    abstention_inferred_count = 0
    
    status_counts = {}
    method_counts = {}
    
    for record in records:
        # Direct abstention field
        if "abstention" in record:
            abstention_count += 1
            if record["abstention"] is True or record["abstention"] == 1:
                abstention_true_count += 1
            elif record["abstention"] is False or record["abstention"] == 0:
                abstention_false_count += 1
        
        # Status field
        if "status" in record:
            status = str(record["status"]).lower()
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Method field
        if "method" in record:
            method = str(record["method"])
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Infer abstention from derivation.abstained
        if "derivation" in record and isinstance(record["derivation"], dict):
            abstained = record["derivation"].get("abstained", 0)
            if abstained > 0:
                abstention_inferred_count += 1
    
    result["abstention_stats"] = {
        "total_cycles": len(records),
        "has_abstention_field": abstention_count,
        "abstention_true": abstention_true_count,
        "abstention_false": abstention_false_count,
        "abstention_rate_direct": abstention_true_count / len(records) if len(records) > 0 else 0.0,
        "abstention_inferred": abstention_inferred_count,
        "abstention_rate_inferred": abstention_inferred_count / len(records) if len(records) > 0 else 0.0,
    }
    
    result["status_distribution"] = status_counts
    result["method_distribution"] = method_counts
    
    return result

def main():
    log_files = [
        Path("results/fo_baseline.jsonl"),
        Path("results/fo_rfl_50.jsonl"),
        Path("results/fo_rfl_1000.jsonl"),
        Path("results/fo_rfl.jsonl"),
    ]
    
    print("=" * 80)
    print("Precise FO Logs Audit (Cursor B)")
    print("=" * 80)
    print()
    
    results = []
    for filepath in log_files:
        result = audit_log(filepath)
        results.append(result)
        
        print(f"File: {filepath.name}")
        if not result["exists"]:
            print("  ❌ NOT FOUND")
        elif result["empty"]:
            print("  ⚠️  EXISTS BUT EMPTY")
        else:
            print(f"  ✅ {result['line_count']} cycles")
            if result["cycle_range"]:
                print(f"     Cycle range: {result['cycle_range'][0]}-{result['cycle_range'][1]}")
                print(f"     Contiguous: {'✅' if result['cycles_contiguous'] else '❌'}")
            print(f"     Schema: {result['schema']}")
            
            stats = result["abstention_stats"]
            print(f"     Abstention (direct): {stats['abstention_true']}/{stats['total_cycles']} = {stats['abstention_rate_direct']:.1%}")
            print(f"     Abstention (inferred): {stats['abstention_inferred']}/{stats['total_cycles']} = {stats['abstention_rate_inferred']:.1%}")
            
            if result["status_distribution"]:
                print(f"     Status distribution: {result['status_distribution']}")
            if result["method_distribution"]:
                print(f"     Method distribution: {result['method_distribution']}")
        print()
    
    # Summary table
    print("=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'File':<25} {'Cycles':<10} {'Schema':<8} {'Abstention Rate':<20} {'Status'}")
    print("-" * 80)
    for result in results:
        if result["exists"] and not result["empty"]:
            file_name = Path(result["file"]).name
            cycles = result["line_count"]
            schema = result["schema"]
            abstention_rate = result["abstention_stats"]["abstention_rate_direct"]
            if abstention_rate == 0.0:
                abstention_rate = result["abstention_stats"]["abstention_rate_inferred"]
            status = "COMPLETE" if result["cycles_contiguous"] and result["cycle_range"] and result["cycle_range"][0] == 0 else "PARTIAL"
            print(f"{file_name:<25} {cycles:<10} {schema:<8} {abstention_rate:>18.1%} {status}")

if __name__ == "__main__":
    main()

