#!/usr/bin/env python3
"""
Quick validation script for FO logs (Cursor B).
"""
import json
import sys
from pathlib import Path

def validate_log(filepath: Path):
    """Validate a single log file."""
    if not filepath.exists():
        return {"exists": False}
    
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if not records:
        return {"exists": True, "empty": True}
    
    cycles = [r.get("cycle") for r in records]
    sample = records[0]
    
    # Compute abstention stats
    abstention_true = sum(1 for r in records if r.get("abstention") is True)
    abstention_inferred = sum(1 for r in records if r.get("derivation", {}).get("abstained", 0) > 0)
    abstention_rate = abstention_true / len(records) if abstention_true > 0 else abstention_inferred / len(records)
    
    # Status/method distribution
    statuses = {}
    methods = {}
    for r in records:
        if "status" in r:
            s = r["status"]
            statuses[s] = statuses.get(s, 0) + 1
        if "method" in r:
            m = r["method"]
            methods[m] = methods.get(m, 0) + 1
    
    return {
        "exists": True,
        "empty": False,
        "line_count": len(records),
        "file_size": filepath.stat().st_size,
        "cycle_range": (min(cycles), max(cycles)),
        "has_slice_name": "slice_name" in sample,
        "has_status": "status" in sample,
        "has_method": "method" in sample,
        "has_abstention": "abstention" in sample,
        "has_derivation_abstained": "derivation" in sample and isinstance(sample.get("derivation"), dict) and "abstained" in sample["derivation"],
        "abstention_rate": abstention_rate,
        "abstention_true": abstention_true,
        "abstention_inferred": abstention_inferred,
        "status_distribution": statuses,
        "method_distribution": methods,
    }

def main():
    files = [
        Path("results/fo_baseline.jsonl"),
        Path("results/fo_rfl_50.jsonl"),
        Path("results/fo_rfl_1000.jsonl"),
        Path("results/fo_rfl.jsonl"),
        Path("results/fo_baseline_wide.jsonl"),
        Path("results/fo_rfl_wide.jsonl"),
    ]
    
    print("=" * 70)
    print("FO Logs Validation (Cursor B)")
    print("=" * 70)
    print()
    
    for filepath in files:
        result = validate_log(filepath)
        print(f"{filepath.name}:")
        if not result.get("exists"):
            print("  ❌ NOT FOUND")
        elif result.get("empty"):
            print("  ⚠️  EXISTS BUT EMPTY")
        else:
            print(f"  ✅ {result['line_count']} cycles, {result['file_size']:,} bytes")
            print(f"     Cycle range: {result['cycle_range'][0]}-{result['cycle_range'][1]}")
            schema = "NEW" if all([result['has_slice_name'], result['has_status'], result['has_method'], result['has_abstention']]) else "OLD"
            print(f"     Schema: {schema}")
            print(f"     Abstention rate: {result['abstention_rate']:.1%} ({result['abstention_true']} direct, {result['abstention_inferred']} inferred)")
            if result['status_distribution']:
                print(f"     Status: {result['status_distribution']}")
            if result['method_distribution']:
                print(f"     Method: {result['method_distribution']}")
            if result['has_derivation_abstained']:
                print(f"     Fallback: derivation.abstained available")
        print()
    
    # Check Dyno chart
    dyno_chart = Path("artifacts/figures/rfl_dyno_chart.png")
    if dyno_chart.exists():
        size = dyno_chart.stat().st_size
        print(f"Dyno Chart: ✅ {dyno_chart} ({size:,} bytes)")
    else:
        print(f"Dyno Chart: ❌ NOT FOUND")

if __name__ == "__main__":
    main()

