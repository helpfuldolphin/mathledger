# Determinism check on H_t
# Usage: Run this from PowerShell at C:\dev\mathledger

$env:FIRST_ORGANISM_TESTS = "false"
$env:MDAP_SEED = "0x4D444150"

uv run python - << 'EOF'
import sys
from pathlib import Path
project_root = str(Path.cwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.run_fo_cycles import CycleRunner

# Create runner in RFL mode
import tempfile
tmp = tempfile.NamedTemporaryFile(delete=True, suffix='.jsonl')
runner = CycleRunner(mode="rfl", output_path=Path(tmp.name), slice_name=None, system="pl")

print("=" * 60)
print("DETERMINISM CHECK: H_t")
print("=" * 60)
print("Running cycle 0 twice with same seed/env...")
print()

# First run
result1 = runner.run_cycle(0)
h_t_1 = result1['roots']['h_t']
r_t_1 = result1['roots']['r_t']
u_t_1 = result1['roots']['u_t']

# Second run (same cycle index, same seed)
result2 = runner.run_cycle(0)
h_t_2 = result2['roots']['h_t']
r_t_2 = result2['roots']['r_t']
u_t_2 = result2['roots']['u_t']

print("First run:")
print(f"  H_t: {h_t_1}")
print(f"  R_t: {r_t_1}")
print(f"  U_t: {u_t_1}")
print()
print("Second run:")
print(f"  H_t: {h_t_2}")
print(f"  R_t: {r_t_2}")
print(f"  U_t: {u_t_2}")
print()
print("-" * 60)

# Check determinism
h_t_match = h_t_1 == h_t_2
r_t_match = r_t_1 == r_t_2
u_t_match = u_t_1 == u_t_2

print("Determinism Results:")
print(f"  H_t match: {h_t_match}")
print(f"  R_t match: {r_t_match}")
print(f"  U_t match: {u_t_match}")
print()

if h_t_match and r_t_match and u_t_match:
    print("✅ FULL DETERMINISM: All roots match")
    print("   MDAP determinism is intact - same inputs → same outputs")
    print("   This is expected and correct for hermetic runs")
else:
    print("❌ NON-DETERMINISM DETECTED")
    print("   Roots differ between runs with same seed")
    print()
    print("   Most likely culprits:")
    if not r_t_match:
        print("   - R_t mismatch: Derivation may have non-deterministic search")
        print("     Check: run_slice_for_test, candidate ordering, hash computation")
    if not u_t_match:
        print("   - U_t mismatch: UI event capture may have timing/ordering issues")
        print("     Check: capture_ui_event, ui_event_store ordering")
    if not h_t_match:
        print("   - H_t mismatch: Composite root depends on R_t and U_t")
        print("     Fix R_t/U_t determinism first")
    print()
    print("   Debug steps:")
    print("   1. Check if derivation uses any random number generation")
    print("   2. Verify all hash functions use deterministic inputs")
    print("   3. Check for any time-based or process-ID based values")
    print("   4. Ensure UI event store is cleared between cycles")
tmp.close()
EOF

