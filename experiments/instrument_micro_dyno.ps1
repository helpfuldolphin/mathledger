# Ten-cycle "micro-Dyno" run
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
print("MICRO-DYNO: 10 RFL CYCLES")
print("=" * 60)
print("Format: cycle | abstain | H_t (first 16 chars)")
print("-" * 60)

h_t_values = []
abstention_count = 0

for i in range(10):
    result = runner.run_cycle(i)
    h_t = result['roots']['h_t']
    h_t_short = h_t[:16] + "..." if len(h_t) > 16 else h_t
    abstain = result['abstention']
    
    if abstain:
        abstention_count += 1
    
    h_t_values.append(h_t)
    
    print(f"{i:3d} | {str(abstain):5s} | H_t: {h_t_short}")

print("-" * 60)
print(f"Summary:")
print(f"  Total cycles: 10")
print(f"  Abstentions: {abstention_count} ({abstention_count*10}%)")
print(f"  Unique H_t values: {len(set(h_t_values))}")
print()
print("Interpretation:")
if abstention_count == 10:
    print("  ⚠️  All cycles abstained - system may be struggling with slice difficulty")
    print("     Expected for hard slices (atoms=7, depth_max=12)")
elif abstention_count > 5:
    print("  ⚠️  High abstention rate - RFL should learn over time")
    print("     Check if abstention decreases in later cycles")
elif abstention_count > 0:
    print("  ✅ Mixed results - some abstentions expected")
    print("     RFL should reduce abstention rate over 1000 cycles")
else:
    print("  ✅ No abstentions - slice may be too easy for RFL uplift measurement")
print()
if len(set(h_t_values)) == 1:
    print("  ⚠️  All H_t values identical - check determinism")
elif len(set(h_t_values)) < 5:
    print("  ⚠️  Low H_t diversity - may indicate limited derivation variation")
else:
    print("  ✅ H_t values vary - good derivation diversity")
tmp.close()
EOF

