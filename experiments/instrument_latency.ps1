# Latency measurement for one RFL cycle
# Usage: Run this from PowerShell at C:\dev\mathledger

$env:FIRST_ORGANISM_TESTS = "false"
$env:MDAP_SEED = "0x4D444150"

uv run python - << 'EOF'
import sys
import time
from pathlib import Path
project_root = str(Path.cwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.run_fo_cycles import CycleRunner

# Create runner in RFL mode
print("Initializing RFL runner...")
import tempfile
tmp = tempfile.NamedTemporaryFile(delete=True, suffix='.jsonl')
init_start = time.time()
runner = CycleRunner(mode="rfl", output_path=Path(tmp.name), slice_name=None, system="pl")
init_time = time.time() - init_start
print(f"Runner initialization: {init_time:.3f}s")
print()

# Measure single cycle latency
print("Running cycle 0...")
cycle_start = time.time()
result = runner.run_cycle(0)
cycle_time = time.time() - cycle_start
tmp.close()

print("=" * 60)
print("LATENCY MEASUREMENT")
print("=" * 60)
print(f"Cycle execution time: {cycle_time:.3f}s")
print(f"Runner init time: {init_time:.3f}s")
print(f"Total time: {init_time + cycle_time:.3f}s")
print()
print("Breakdown (estimated):")
print("  - Derivation: ~60-80% of cycle time (if high, check run_slice_for_test)")
print("  - Attestation (H_t): ~10-15% (seal_block_with_dual_roots)")
print("  - RFL metabolism: ~10-20% (if executed)")
print("  - Gate evaluation: <5%")
print()
if cycle_time > 1.0:
    print("⚠️  WARNING: Cycle time > 1s")
    print("   Most likely culprits:")
    print("   1. Derivation (run_slice_for_test) - check Lean interface calls")
    print("   2. RFL runner initialization or policy updates")
    print("   3. Heavy logging or file I/O")
elif cycle_time > 0.5:
    print("⚠️  Cycle time > 0.5s - monitor for consistency")
else:
    print("✅ Cycle time < 0.5s - healthy")
print()
print(f"Result: status={result['status']}, abstention={result['abstention']}")
EOF

