#!/usr/bin/env python
"""Check candidate counts from experiment results."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/test_no_stop.jsonl"
data = [json.loads(l) for l in open(path) if l.strip()]

print("Candidates per cycle (first 10):")
for i, d in enumerate(data[:10]):
    deriv = d.get("derivation", {})
    print(f"  Cycle {i}: candidates={deriv.get('candidates', 0)}, "
          f"verified={deriv.get('verified', 0)}, abstained={deriv.get('abstained', 0)}")

