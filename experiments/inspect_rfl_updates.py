#!/usr/bin/env python
"""Inspect RFL policy updates from JSONL logs."""
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

def analyze_rfl_log(path: str) -> None:
    """Analyze RFL log to check if policy is updating."""
    data: List[Dict[str, Any]] = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    
    if not data:
        print(f"{path}: No data found")
        return
    
    # Extract RFL stats
    rfl_cycles = [d for d in data if d.get('rfl', {}).get('executed', False)]
    
    if not rfl_cycles:
        print(f"{path}: No RFL cycles found (all cycles may be baseline mode)")
        return
    
    print(f"\n{path}:")
    print(f"  Total cycles: {len(data)}")
    print(f"  RFL cycles: {len(rfl_cycles)}")
    
    # Check policy update progression
    update_counts = [d['rfl'].get('update_count', 0) for d in rfl_cycles]
    policy_updates = [d['rfl'].get('policy_update', False) for d in rfl_cycles]
    ledger_lengths = [d['rfl'].get('policy_ledger_length', 0) for d in rfl_cycles]
    
    print(f"\n  Policy Update Analysis:")
    print(f"    Cycles with updates: {sum(policy_updates)}/{len(policy_updates)}")
    print(f"    Final update_count: {update_counts[-1] if update_counts else 0}")
    print(f"    Final ledger_length: {ledger_lengths[-1] if ledger_lengths else 0}")
    
    # Show progression
    if len(update_counts) > 1:
        print(f"\n  Update Count Progression (first 10, last 10):")
        first_10 = update_counts[:10]
        last_10 = update_counts[-10:]
        print(f"    First 10:  {first_10}")
        print(f"    Last 10:   {last_10}")
        
        if update_counts[-1] > update_counts[0]:
            print(f"    ✓ Policy IS updating: {update_counts[0]} → {update_counts[-1]}")
        else:
            print(f"    ✗ Policy NOT updating: stuck at {update_counts[-1]}")
    
    # Check symbolic descent values
    symbolic_descents = [d['rfl'].get('symbolic_descent', 0.0) for d in rfl_cycles]
    if symbolic_descents:
        print(f"\n  Symbolic Descent Stats:")
        print(f"    Mean: {sum(symbolic_descents)/len(symbolic_descents):.6f}")
        print(f"    Min:  {min(symbolic_descents):.6f}")
        print(f"    Max:  {max(symbolic_descents):.6f}")
        print(f"    Non-zero: {sum(1 for sd in symbolic_descents if abs(sd) > 1e-9)}/{len(symbolic_descents)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python inspect_rfl_updates.py <path-to-rfl-jsonl>")
        sys.exit(1)
    analyze_rfl_log(sys.argv[1])

