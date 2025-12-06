#!/usr/bin/env python
"""Check policy weight changes."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'results/uplift_proto_rfl_test2.jsonl'

data = []
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

if len(data) < 2:
    print(f"Not enough data: {len(data)} cycles")
    sys.exit(1)

w0 = data[0].get('policy_weights', {})
w_last = data[-1].get('policy_weights', {})

print(f"Policy weights over {len(data)} cycles:")
print(f"  Cycle 0:  len={w0.get('len', 0):.4f}, depth={w0.get('depth', 0):.4f}")
print(f"  Cycle {len(data)-1}: len={w_last.get('len', 0):.4f}, depth={w_last.get('depth', 0):.4f}")
print(f"  Change:   len={w_last.get('len', 0) - w0.get('len', 0):+.4f}, depth={w_last.get('depth', 0) - w0.get('depth', 0):+.4f}")

