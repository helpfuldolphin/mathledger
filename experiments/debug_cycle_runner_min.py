#!/usr/bin/env python3
"""Minimal CycleRunner test with known-good slice."""

from pathlib import Path
import sys

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.run_fo_cycles import CycleRunner

def main():
    out = Path("results/debug_min_pl.jsonl")
    print(f"[DEBUG] Starting CycleRunner.run(3) → {out}", flush=True)
    try:
        runner = CycleRunner(
            mode="baseline",
            output_path=out,
            slice_name="slice_easy_fo",  # known good from FO
            system="pl",
        )
        print(f"[DEBUG] CycleRunner created, calling run(3)...", flush=True)
        runner.run(3)
        print(f"[DEBUG] Done. Exists={out.exists()}, size={out.stat().st_size if out.exists() else 'N/A'}", flush=True)
        if out.exists() and out.stat().st_size > 0:
            print(f"[DEBUG] First line:", flush=True)
            with open(out) as f:
                first_line = f.readline()
                print(f"  {first_line.rstrip()}", flush=True)
    except Exception as e:
        print(f"[DEBUG] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

