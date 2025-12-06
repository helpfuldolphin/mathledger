#!/usr/bin/env python
import json
import sys
from collections import Counter


def main(path: str) -> None:
    total = 0
    abstain = 0
    hashes = Counter()
    slices = Counter()
    methods = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)

            # schema-tolerant abstention detection
            if rec.get("abstention") is True:
                abstain += 1
            elif rec.get("status") == "abstain":
                abstain += 1
            elif rec.get("derivation", {}).get("abstained", 0) > 0:
                abstain += 1

            cand_hash = rec.get("derivation", {}).get("candidate_hash")
            if cand_hash:
                hashes[cand_hash] += 1

            slices[rec.get("slice_name", "<none>")] += 1
            methods[rec.get("method", "<none>")] += 1

    print(f"File: {path}")
    print(f"  total cycles:         {total}")
    print(f"  abstentions:          {abstain}")
    print(f"  abstention rate:      {abstain / total if total else 0:.3f}")
    print(f"  unique candidate_hash: {len(hashes)}")
    if hashes:
        most_common = hashes.most_common(5)
        print("  top 5 candidate_hash frequencies:")
        for h, c in most_common:
            print(f"    {h[:12]}… : {c}")
    print("  slice_name counts:", dict(slices))
    print("  method counts:    ", dict(methods))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python check_fo_diversity.py <path-to-jsonl>")
        sys.exit(1)
    main(sys.argv[1])

