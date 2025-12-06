#!/usr/bin/env python3
"""Create provenance manifest directly from log analysis."""

import json
from pathlib import Path
from datetime import datetime

# Analyze logs
baseline_path = Path("results/fo_baseline.jsonl")
rfl_path = Path("results/fo_rfl.jsonl")

baseline_records = []
with open(baseline_path) as f:
    for line in f:
        if line.strip():
            baseline_records.append(json.loads(line))

rfl_records = []
with open(rfl_path) as f:
    for line in f:
        if line.strip():
            rfl_records.append(json.loads(line))

# Extract metadata
baseline_cycles = len(baseline_records)
rfl_cycles = len(rfl_records)

baseline_cycle_range = [
    min(e.get('cycle', 0) for e in baseline_records),
    max(e.get('cycle', 0) for e in baseline_records)
]

rfl_cycle_range = [
    min(e.get('cycle', 0) for e in rfl_records),
    max(e.get('cycle', 0) for e in rfl_records)
]

slice_name = rfl_records[0].get('slice_name', 'first-organism-pl') if rfl_records else None
slice_display = 'Default Slice'  # first-organism-pl is the default slice

# Calculate abstention rates
baseline_abstentions = sum(1 for e in baseline_records if e.get('derivation', {}).get('abstained', 0) > 0)
rfl_abstentions = sum(1 for e in rfl_records if e.get('abstention', False) or e.get('status') == 'abstain')

baseline_abstention_rate = baseline_abstentions / baseline_cycles if baseline_cycles > 0 else 0
rfl_abstention_rate = rfl_abstentions / rfl_cycles if rfl_cycles > 0 else 0

manifest = {
    'chart_name': 'rfl_dyno_chart',
    'generated_at': datetime.now().isoformat(),
    'baseline_path': str(baseline_path.resolve()),
    'rfl_path': str(rfl_path.resolve()),
    'baseline_cycles': baseline_cycles,
    'rfl_cycles': rfl_cycles,
    'baseline_cycle_range': baseline_cycle_range,
    'rfl_cycle_range': rfl_cycle_range,
    'window_size': 100,
    'slice_name': slice_name,
    'slice_display': slice_display,
    'abstention_summary': {
        'baseline_total_abstentions': baseline_abstentions,
        'baseline_abstention_rate': round(baseline_abstention_rate, 4),
        'rfl_total_abstentions': rfl_abstentions,
        'rfl_abstention_rate': round(rfl_abstention_rate, 4),
        'uplift_demonstrated': False,
        'note': 'Both series show flat abstention (100%). This validates the plotting pipeline but does not demonstrate RFL uplift.'
    },
    'output_files': {
        'png': 'artifacts/figures/rfl_dyno_chart.png',
        'pdf': 'artifacts/figures/rfl_dyno_chart.pdf',
        'manifest': 'artifacts/figures/rfl_dyno_chart_manifest.json'
    }
}

# Save manifest
manifest_path = Path('artifacts/figures/rfl_dyno_chart_manifest.json')
manifest_path.parent.mkdir(parents=True, exist_ok=True)

with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"Manifest created: {manifest_path}")
print(f"\nSummary:")
print(f"  Baseline: {baseline_cycles} cycles, {baseline_abstentions} abstentions ({baseline_abstention_rate*100:.1f}%)")
print(f"  RFL: {rfl_cycles} cycles, {rfl_abstentions} abstentions ({rfl_abstention_rate*100:.1f}%)")
print(f"  Slice: {slice_display}")

