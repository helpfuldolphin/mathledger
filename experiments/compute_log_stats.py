#!/usr/bin/env python3
"""Quick stats computation for FO logs."""
import json
from pathlib import Path

files = {
    'fo_baseline.jsonl': 'results/fo_baseline.jsonl',
    'fo_rfl_50.jsonl': 'results/fo_rfl_50.jsonl',
    'fo_rfl_1000.jsonl': 'results/fo_rfl_1000.jsonl',
    'fo_rfl.jsonl': 'results/fo_rfl.jsonl',
}

for name, path in files.items():
    p = Path(path)
    if not p.exists():
        print(f"{name}: NOT FOUND")
        continue
    
    records = [json.loads(l) for l in open(p) if l.strip()]
    if not records:
        print(f"{name}: EMPTY")
        continue
    
    cycles = [r['cycle'] for r in records]
    min_cycle, max_cycle = min(cycles), max(cycles)
    expected = set(range(min_cycle, max_cycle + 1))
    actual = set(cycles)
    contiguous = (expected == actual and len(cycles) == len(expected))
    
    # Schema
    sample = records[0]
    schema = "NEW" if all(k in sample for k in ['slice_name', 'status', 'method', 'abstention']) else "OLD"
    
    # Abstention stats
    abstention_true = sum(1 for r in records if r.get('abstention') is True)
    abstention_inferred = sum(1 for r in records if r.get('derivation', {}).get('abstained', 0) > 0)
    abstention_rate = abstention_true / len(records) if abstention_true > 0 else abstention_inferred / len(records)
    
    # Status/method distribution
    statuses = {}
    methods = {}
    for r in records:
        if 'status' in r:
            s = r['status']
            statuses[s] = statuses.get(s, 0) + 1
        if 'method' in r:
            m = r['method']
            methods[m] = methods.get(m, 0) + 1
    
    status_str = ', '.join(f"{k}:{v}" for k, v in statuses.items()) if statuses else "N/A"
    method_str = ', '.join(f"{k}:{v}" for k, v in methods.items()) if methods else "N/A"
    
    print(f"{name}:")
    print(f"  Lines: {len(records)}")
    print(f"  Cycles: {min_cycle}-{max_cycle} ({'contiguous' if contiguous else 'gaps'})")
    print(f"  Schema: {schema}")
    print(f"  Abstention rate: {abstention_rate:.1%} ({abstention_true} direct, {abstention_inferred} inferred)")
    print(f"  Status: {status_str}")
    print(f"  Method: {method_str}")
    print()

