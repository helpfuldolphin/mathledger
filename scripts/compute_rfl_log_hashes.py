#!/usr/bin/env python3
"""Compute SHA256 hashes for RFL log files."""

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

files = {
    "fo_rfl_50": REPO_ROOT / "results" / "fo_rfl_50.jsonl",
    "fo_rfl": REPO_ROOT / "results" / "fo_rfl.jsonl",
    "fo_rfl_1000": REPO_ROOT / "results" / "fo_rfl_1000.jsonl",
}

results = {}
for name, path in files.items():
    if path.exists():
        with open(path, 'rb') as f:
            content = f.read()
            sha256 = hashlib.sha256(content).hexdigest()
            size = len(content)
            lines = [l for l in content.decode('utf-8').split('\n') if l.strip()]
            cycles = [json.loads(l)['cycle'] for l in lines if l.strip()]
            results[name] = {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256,
                "size_bytes": size,
                "line_count": len(lines),
                "cycle_count": len(cycles),
                "first_cycle": min(cycles) if cycles else None,
                "last_cycle": max(cycles) if cycles else None,
            }
    else:
        results[name] = {"exists": False}

print(json.dumps(results, indent=2))

