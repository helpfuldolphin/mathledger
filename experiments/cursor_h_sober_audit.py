#!/usr/bin/env python3
"""
Cursor H - Dyno Chart Sober Truth Audit
========================================

In Reviewer 2 mode: Verify what actually exists, what the Dyno Chart is based on,
and document inconsistencies between claims and reality.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Key files to check
BASELINE_FILES = [
    "results/fo_baseline.jsonl",
    "results/fo_baseline_wide.jsonl",
]

RFL_FILES = [
    "results/fo_rfl.jsonl",
    "results/fo_rfl_50.jsonl",
    "results/fo_rfl_1000.jsonl",
    "results/fo_rfl_wide.jsonl",
]

DYNO_CHART = Path("artifacts/figures/rfl_dyno_chart.png")
ABSTENTION_CHART = Path("artifacts/figures/rfl_abstention_rate.png")

def check_file(path: str) -> Dict[str, Any]:
    """Check if file exists and get metadata."""
    p = Path(path)
    result = {
        "path": path,
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else 0,
        "lines": 0,
        "cycles": [],
        "sample_entry": None,
    }
    
    if p.exists() and p.stat().st_size > 0:
        try:
            with open(p, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                result["lines"] = len(lines)
                
                if lines:
                    # Parse first entry
                    try:
                        sample = json.loads(lines[0])
                        result["sample_entry"] = sample
                        if "cycle" in sample:
                            result["cycles"].append(sample["cycle"])
                    except:
                        pass
                    
                    # Parse last entry to get max cycle
                    if len(lines) > 1:
                        try:
                            last = json.loads(lines[-1])
                            if "cycle" in last:
                                result["cycles"].append(last["cycle"])
                        except:
                            pass
        except Exception as e:
            result["error"] = str(e)
    
    return result

def main():
    print("=" * 80)
    print("CURSOR H - DYNO CHART SOBER TRUTH AUDIT")
    print("=" * 80)
    print()
    
    print("1. BASELINE LOG FILES")
    print("-" * 80)
    baseline_data = {}
    for f in BASELINE_FILES:
        data = check_file(f)
        baseline_data[f] = data
        status = "✓ EXISTS" if data["exists"] and data["size_bytes"] > 0 else "✗ MISSING/EMPTY"
        print(f"  {status}: {f}")
        if data["exists"]:
            print(f"    Size: {data['size_bytes']:,} bytes")
            print(f"    Lines: {data['lines']:,}")
            if data["cycles"]:
                print(f"    Cycle range: {min(data['cycles'])} - {max(data['cycles'])}")
    
    print()
    print("2. RFL LOG FILES")
    print("-" * 80)
    rfl_data = {}
    for f in RFL_FILES:
        data = check_file(f)
        rfl_data[f] = data
        status = "✓ EXISTS" if data["exists"] and data["size_bytes"] > 0 else "✗ MISSING/EMPTY"
        print(f"  {status}: {f}")
        if data["exists"]:
            print(f"    Size: {data['size_bytes']:,} bytes")
            print(f"    Lines: {data['lines']:,}")
            if data["cycles"]:
                print(f"    Cycle range: {min(data['cycles'])} - {max(data['cycles'])}")
    
    print()
    print("3. DYNO CHART FIGURES")
    print("-" * 80)
    print(f"  {'✓ EXISTS' if DYNO_CHART.exists() else '✗ MISSING'}: {DYNO_CHART}")
    if DYNO_CHART.exists():
        size = DYNO_CHART.stat().st_size
        print(f"    Size: {size:,} bytes")
        if size == 0:
            print("    ⚠️  WARNING: File exists but is 0 bytes (invalid)")
    
    print(f"  {'✓ EXISTS' if ABSTENTION_CHART.exists() else '✗ MISSING'}: {ABSTENTION_CHART}")
    if ABSTENTION_CHART.exists():
        size = ABSTENTION_CHART.stat().st_size
        print(f"    Size: {size:,} bytes")
    
    print()
    print("4. SOBER TRUTH ASSESSMENT")
    print("-" * 80)
    
    # Find viable baseline
    viable_baseline = None
    for f, data in baseline_data.items():
        if data["exists"] and data["size_bytes"] > 0 and data["lines"] > 0:
            viable_baseline = (f, data)
            break
    
    # Find viable RFL
    viable_rfl = None
    for f, data in rfl_data.items():
        if data["exists"] and data["size_bytes"] > 0 and data["lines"] > 0:
            viable_rfl = (f, data)
            break
    
    if viable_baseline and viable_rfl:
        print(f"  ✓ Viable baseline log: {viable_baseline[0]} ({viable_baseline[1]['lines']} cycles)")
        print(f"  ✓ Viable RFL log: {viable_rfl[0]} ({viable_rfl[1]['lines']} cycles)")
        
        if DYNO_CHART.exists() and DYNO_CHART.stat().st_size > 0:
            print(f"  ✓ Dyno Chart exists and is non-empty")
            print()
            print("  RECOMMENDATION: Dyno Chart CAN be generated/regenerated with:")
            print(f"    --baseline {viable_baseline[0]}")
            print(f"    --rfl {viable_rfl[0]}")
        else:
            print(f"  ✗ Dyno Chart missing or empty")
            print()
            print("  RECOMMENDATION: Generate Dyno Chart with:")
            print(f"    --baseline {viable_baseline[0]}")
            print(f"    --rfl {viable_rfl[0]}")
    else:
        print("  ✗ No viable baseline+RFL pair found for Dyno Chart")
        if not viable_baseline:
            print("    - No valid baseline log found")
        if not viable_rfl:
            print("    - No valid RFL log found")
    
    print()
    print("5. WIDE SLICE STATUS")
    print("-" * 80)
    wide_baseline = baseline_data.get("results/fo_baseline_wide.jsonl", {})
    wide_rfl = rfl_data.get("results/fo_rfl_wide.jsonl", {})
    
    if wide_baseline.get("exists") and wide_rfl.get("exists"):
        print("  ✓ Wide Slice logs exist (slice_medium)")
    else:
        print("  ✗ Wide Slice logs NOT YET GENERATED")
        print("    These require: --slice-name=slice_medium --system=pl")
        print("    Status: PLANNED / NOT YET RUN (Phase II)")
    
    print()
    print("=" * 80)
    print("END OF AUDIT")
    print("=" * 80)

if __name__ == "__main__":
    main()

