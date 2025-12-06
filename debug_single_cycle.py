import os
import sys
import json
from pathlib import Path
import tempfile

# Make sure project root is importable
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 60)
print("SINGLE CYCLE RFL PROBE (standalone, no DB/Redis)")
print("=" * 60)

# Environment sanity
print(f"FIRST_ORGANISM_TESTS = {os.environ.get('FIRST_ORGANISM_TESTS')!r}")
print(f"MDAP_SEED            = {os.environ.get('MDAP_SEED')!r}")
print()

from experiments.run_fo_cycles import CycleRunner

# Temporary output file just to satisfy the runner
tmp = tempfile.NamedTemporaryFile(delete=True, suffix=".jsonl")
try:
    runner = CycleRunner(
        mode="rfl",
        output_path=Path(tmp.name),
        slice_name=None,
        system="pl",
    )
    print("Runner created. Running cycle 0...\n")

    result = runner.run_cycle(0)

    print("=" * 60)
    print("SINGLE CYCLE RESULT (summary)")
    print("=" * 60)
    print(f"Cycle:       {result.get('cycle')}")
    print(f"Mode:        {result.get('mode')}")
    print(f"Slice:       {result.get('slice_name')}")
    print(f"Status:      {result.get('status')}")
    print(f"Method:      {result.get('method')}")
    print(f"Abstention:  {result.get('abstention')}")
    print(f"GatesPassed: {result.get('gates_passed')}")
    print()

    roots = result.get("roots", {}) or {}
    print("Roots:")
    print(f"  H_t: {roots.get('h_t')}")
    print(f"  R_t: {roots.get('r_t')}")
    print(f"  U_t: {roots.get('u_t')}")
    print()

    der = result.get("derivation", {}) or {}
    print("Derivation:")
    print(f"  candidates:    {der.get('candidates')}")
    print(f"  abstained:     {der.get('abstained')}")
    print(f"  verified:      {der.get('verified')}")
    print(f"  candidate_hash:{der.get('candidate_hash')}")
    print()

    rfl = result.get("rfl", {}) or {}
    print("RFL Stats:")
    print(f"  executed:              {rfl.get('executed')}")
    print(f"  policy_update:         {rfl.get('policy_update')}")
    print(f"  symbolic_descent:      {rfl.get('symbolic_descent')}")
    print(f"  abstention_rate_before:{rfl.get('abstention_rate_before')}")
    print()

    print("=" * 60)
    print("FULL JSON RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2, sort_keys=True))

finally:
    tmp.close()
