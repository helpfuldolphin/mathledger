import os
import sys
from pathlib import Path
import tempfile

project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.run_fo_cycles import CycleRunner

print("=" * 60)
print("MICRO-DYNO: 10 RFL CYCLES (standalone, no DB/Redis)")
print("=" * 60)
print("Format: cycle | abstain | H_t (first 16 chars)")
print("-" * 60)

os.environ["FIRST_ORGANISM_TESTS"] = "false"
os.environ.setdefault("MDAP_SEED", "0x4D444150")

tmp = tempfile.NamedTemporaryFile(delete=True, suffix=".jsonl")
runner = CycleRunner(mode="rfl", output_path=Path(tmp.name), slice_name=None, system="pl")

h_t_values = []
abstention_count = 0

for i in range(10):
    result = runner.run_cycle(i)
    h_t = result["roots"]["h_t"]
    h_t_short = (h_t[:16] + "...") if len(h_t) > 16 else h_t
    abstain = result["abstention"]

    if abstain:
        abstention_count += 1

    h_t_values.append(h_t)
    print(f"{i:3d} | {str(abstain):5s} | H_t: {h_t_short}")

print("-" * 60)
print(f"Summary:")
print(f"  Total cycles:      10")
print(f"  Abstentions:       {abstention_count} ({abstention_count*10}%)")
print(f"  Unique H_t values: {len(set(h_t_values))}")
print()

if abstention_count == 10:
    print("  ⚠️ All cycles abstained - slice may be very hard (expected for 'slice_hard').")
elif abstention_count > 5:
    print("  ⚠️ High abstention rate - this is the burn-in regime RFL is supposed to improve from.")
elif abstention_count > 0:
    print("  ✅ Mixed results - good for seeing uplift in a longer run.")
else:
    print("  ✅ No abstentions - slice may be too easy to measure RFL uplift.")
print()

if len(set(h_t_values)) == 1:
    print("  ⚠️ All H_t values identical - check determinism / diversity.")
elif len(set(h_t_values)) < 5:
    print("  ⚠️ Low H_t diversity - derivation may be stuck on a narrow pattern.")
else:
    print("  ✅ H_t values vary - good derivation diversity.")

tmp.close()

