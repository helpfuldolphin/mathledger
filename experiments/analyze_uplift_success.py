#!/usr/bin/env python
"""Quick analysis script for uplift experiment results."""
import json
import sys
from pathlib import Path

def main(path: str) -> None:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    
    total = len(data)
    successes = sum(1 for d in data if d.get('success', False))
    verified_counts = [d.get('derivation', {}).get('verified', 0) for d in data]
    
    print(f"File: {path}")
    print(f"  Total cycles:         {total}")
    print(f"  Successful cycles:    {successes}")
    print(f"  Success rate:         {successes/total:.3f}" if total > 0 else "  Success rate:         0.000")
    print(f"  Verified proofs:      {verified_counts}")
    print(f"  Avg verified/cycle:   {sum(verified_counts)/total:.2f}" if total > 0 else "  Avg verified/cycle:   0.00")
    
    # Show status breakdown
    status_counts = {}
    for d in data:
        status = d.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"  Status breakdown:     {dict(status_counts)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python analyze_uplift_success.py <path-to-jsonl>")
        sys.exit(1)
    main(sys.argv[1])

