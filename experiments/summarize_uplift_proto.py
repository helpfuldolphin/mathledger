#!/usr/bin/env python
"""Summarize uplift proto experiment results."""
import json
import sys
from pathlib import Path

def main(path: str) -> None:
    total = success = 0
    verifieds = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1
            if rec.get("success"):
                success += 1
            deriv = rec.get("derivation", {})
            verifieds.append(deriv.get("verified", 0))
    
    success_rate = success / total if total > 0 else 0.0
    avg_verified = sum(verifieds) / len(verifieds) if verifieds else 0.0
    
    print(f"File: {path}")
    print(f"Cycles: {total}")
    print(f"Success: {success} ({success_rate:.3f})")
    print(f"Avg verified: {avg_verified:.2f}")
    print(f"Verified counts (first 10): {verifieds[:10]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python summarize_uplift_proto.py <path-to-jsonl>")
        sys.exit(1)
    main(sys.argv[1])

