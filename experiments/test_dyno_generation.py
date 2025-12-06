#!/usr/bin/env python3
"""Test Dyno Chart generation with actual Phase I logs."""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.plotting import make_dyno_chart, save_figure

def main():
    baseline_path = "results/fo_baseline.jsonl"
    rfl_path = "results/fo_rfl.jsonl"
    output_name = "rfl_dyno_chart"
    window = 100
    
    print(f"Generating Dyno Chart...")
    print(f"  Baseline: {baseline_path}")
    print(f"  RFL: {rfl_path}")
    print(f"  Window: {window}")
    
    # Generate chart with metadata
    fig, metadata = make_dyno_chart(
        baseline_path=baseline_path,
        rfl_path=rfl_path,
        window=window,
        return_metadata=True
    )
    
    print(f"\nMetadata collected:")
    print(f"  Baseline cycles: {metadata['baseline_cycles']}")
    print(f"  RFL cycles: {metadata['rfl_cycles']}")
    print(f"  Slice: {metadata['slice_display']}")
    
    # Save figure
    saved_path = save_figure(output_name, fig)
    print(f"\nChart saved to: {saved_path}")
    
    # Generate manifest
    manifest = {
        'chart_name': output_name,
        'generated_at': datetime.now().isoformat(),
        'baseline_path': str(Path(baseline_path).resolve()),
        'rfl_path': str(Path(rfl_path).resolve()),
        'baseline_cycles': metadata['baseline_cycles'],
        'rfl_cycles': metadata['rfl_cycles'],
        'baseline_cycle_range': metadata['baseline_cycle_range'],
        'rfl_cycle_range': metadata['rfl_cycle_range'],
        'window_size': metadata['window_size'],
        'slice_name': metadata['slice_name'],
        'slice_display': metadata['slice_display'],
        'output_files': {
            'png': str(Path(saved_path).resolve()),
            'pdf': str(Path('artifacts/figures') / f"{output_name}.pdf")
        }
    }
    
    manifest_path = Path('artifacts/figures') / f"{output_name}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest saved to: {manifest_path}")
    print(f"\n✓ Complete!")

if __name__ == "__main__":
    main()

