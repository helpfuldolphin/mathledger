#!/usr/bin/env python
"""Compare candidate ordering between baseline and RFL runs."""
import json
import sys
import re
from typing import Dict, List, Any, Optional

def extract_debug_logs(path: str) -> List[Dict[str, Any]]:
    """Extract CANDIDATE_ORDERING_DEBUG logs from stderr output."""
    debug_logs = []
    
    # Read the file and look for CANDIDATE_ORDERING_DEBUG lines
    # These would be in stderr, but for now we'll check if they're in the JSONL
    # or we need to capture them separately
    
    # For now, let's assume we need to run with DEBUG_CANDIDATE_ORDERING=1
    # and capture stderr separately, or add the debug info to the JSONL
    return debug_logs

def compare_cycles(baseline_path: str, rfl_path: str, cycle_idx: int = 0) -> None:
    """Compare candidate ordering for a specific cycle."""
    print(f"Comparing cycle {cycle_idx} between baseline and RFL")
    print("=" * 70)
    print()
    print("NOTE: This requires running with DEBUG_CANDIDATE_ORDERING=1")
    print("      to capture candidate ordering debug logs.")
    print()
    print("To enable:")
    print("  $env:DEBUG_CANDIDATE_ORDERING='1'")
    print("  uv run python experiments/run_fo_cycles.py ...")
    print()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python compare_candidate_ordering.py <baseline.jsonl> <rfl.jsonl> [cycle_idx]")
        sys.exit(1)
    
    cycle_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    compare_cycles(sys.argv[1], sys.argv[2], cycle_idx)

