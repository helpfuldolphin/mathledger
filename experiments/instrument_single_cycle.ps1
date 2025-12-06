# Single-cycle RFL probe
# Usage: Run this from PowerShell at C:\dev\mathledger
# Prerequisites: NO DATABASE_URL or REDIS_URL set

$env:FIRST_ORGANISM_TESTS = "false"
$env:MDAP_SEED = "0x4D444150"

uv run python - << 'EOF'
import sys
from pathlib import Path
project_root = str(Path.cwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.run_fo_cycles import CycleRunner
import json

# Create runner in RFL mode (output_path not used for single cycle)
import tempfile
tmp = tempfile.NamedTemporaryFile(delete=True, suffix='.jsonl')
runner = CycleRunner(mode="rfl", output_path=Path(tmp.name), slice_name=None, system="pl")

# Run single cycle
result = runner.run_cycle(0)
tmp.close()

# Print readable output
print("=" * 60)
print("SINGLE CYCLE RFL PROBE")
print("=" * 60)
print(f"Cycle: {result['cycle']}")
print(f"Mode: {result['mode']}")
print(f"Slice: {result['slice_name']}")
print(f"Status: {result['status']}")
print(f"Method: {result['method']}")
print(f"Abstention: {result['abstention']}")
print(f"Gates Passed: {result['gates_passed']}")
print()
print("Roots:")
print(f"  H_t: {result['roots']['h_t']}")
print(f"  R_t: {result['roots']['r_t']}")
print(f"  U_t: {result['roots']['u_t']}")
print()
print("Derivation:")
print(f"  Candidates: {result['derivation']['candidates']}")
print(f"  Abstained: {result['derivation']['abstained']}")
print(f"  Verified: {result['derivation']['verified']}")
print(f"  Candidate Hash: {result['derivation']['candidate_hash']}")
print()
print("RFL Stats:")
rfl = result['rfl']
print(f"  Executed: {rfl['executed']}")
if rfl['executed']:
    print(f"  Policy Update Applied: {rfl.get('policy_update', False)}")
    print(f"  Symbolic Descent: {rfl.get('symbolic_descent', 'N/A')}")
    print(f"  Abstention Rate Before: {rfl.get('abstention_rate_before', 'N/A')}")
print()
print("Full JSON (for inspection):")
print(json.dumps(result, indent=2, sort_keys=True))
EOF

