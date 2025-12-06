#!/usr/bin/env python
"""Quick summarizer for debug uplift experiment results."""
import json
import sys
from pathlib import Path

def main(path: str) -> None:
    total = succ = 0
    verified_counts = []
    
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1
            if rec.get("success"):
                succ += 1
            verified = rec.get("derivation", {}).get("verified", 0)
            verified_counts.append(verified)
    
    success_rate = succ / total if total > 0 else 0.0
    avg_verified = sum(verified_counts) / len(verified_counts) if verified_counts else 0.0
    
    print(f"{path}: {succ}/{total} success = {success_rate:.3f}")
    print(f"  Avg verified proofs per cycle: {avg_verified:.2f}")
    print(f"  Verified counts: {verified_counts[:10]}{'...' if len(verified_counts) > 10 else ''}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python summarize_debug_uplift.py <path-to-jsonl>")
        sys.exit(1)
    main(sys.argv[1])

