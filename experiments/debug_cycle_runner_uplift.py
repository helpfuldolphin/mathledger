#!/usr/bin/env python3
"""Debug script to test CycleRunner directly with slice_pl_uplift_a."""

from pathlib import Path
import sys

# Make sure project root is on path
from pathlib import Path as _P
project_root = str(_P(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.run_fo_cycles import CycleRunner

def main():
    import traceback
    out = Path("results/debug_uplift_baseline.jsonl")
    print(f"[DEBUG] Starting debug script...", flush=True)
    print(f"[DEBUG] Output path: {out}", flush=True)
    
    try:
        print(f"[DEBUG] Step 1: Importing CycleRunner...", flush=True)
        from experiments.run_fo_cycles import CycleRunner
        print(f"[DEBUG] Step 2: CycleRunner imported", flush=True)
        
        print(f"[DEBUG] Step 3: Creating CycleRunner with slice_pl_uplift_a...", flush=True)
        runner = CycleRunner(
            mode="baseline",
            output_path=out,
            slice_name="slice_pl_uplift_a",
            system="pl",
        )
        print(f"[DEBUG] Step 4: CycleRunner created successfully", flush=True)
        print(f"[DEBUG] Step 5: Running 1 cycle → {out}", flush=True)
        
        # Try running just 1 cycle first
        runner.run(1)
        
        print(f"[DEBUG] Step 6: Run complete", flush=True)
        print(f"[DEBUG] File exists: {out.exists()}", flush=True)
        if out.exists():
            size = out.stat().st_size
            print(f"[DEBUG] File size: {size} bytes", flush=True)
            if size > 0:
                print(f"[DEBUG] First few lines:", flush=True)
                with open(out) as f:
                    for i, line in enumerate(f):
                        if i >= 3:
                            break
                        print(f"  {line.rstrip()}", flush=True)
            else:
                print(f"[DEBUG] WARNING: File exists but is empty!", flush=True)
        else:
            print(f"[DEBUG] ERROR: File was not created!", flush=True)
    except Exception as e:
        print(f"[DEBUG] EXCEPTION: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise

if __name__ == "__main__":
    main()

