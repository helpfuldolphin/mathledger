#!/usr/bin/env python3
"""
Standalone verification script for MathLedger drop-in demo.

This script can be run independently to verify:
1. Composite root integrity: H_t == SHA256(R_t || U_t)
2. Reproducibility: re-running demo produces identical outputs

Usage:
    python verify.py

Expected output:
    [PASS] Composite root verified
    [INFO] To verify reproducibility, run the demo again with seed 42
"""

import hashlib
import json
from pathlib import Path

def main():
    # Load roots
    r_t = Path("reasoning_root.txt").read_text().strip()
    u_t = Path("ui_root.txt").read_text().strip()
    h_t = Path("epoch_root.txt").read_text().strip()

    # Verify composite
    computed = hashlib.sha256((r_t + u_t).encode("ascii")).hexdigest()

    if computed == h_t:
        print("[PASS] Composite root verified: H_t == SHA256(R_t || U_t)")
    else:
        print("[FAIL] Composite root mismatch!")
        print(f"  Expected: {h_t}")
        print(f"  Computed: {computed}")
        exit(1)

    # Load manifest
    manifest = json.loads(Path("manifest.json").read_text())
    print(f"[INFO] Seed: {manifest['seed']}")
    print(f"[INFO] Claim level: {manifest['governance']['claim_level']}")
    print(f"[INFO] F5 codes: {manifest['governance']['f5_codes']}")
    print(f"[INFO] To verify reproducibility, run: uv run python scripts/run_dropin_demo.py --seed 42 --output demo_output_verify/")

if __name__ == "__main__":
    main()
