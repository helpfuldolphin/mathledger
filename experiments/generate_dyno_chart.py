"""
Dyno Chart Generator CLI
========================

Command-line interface for generating the RFL Dyno Chart from Wide Slice logs.

Usage:
    uv run python experiments/generate_dyno_chart.py \
        --baseline results/fo_baseline_wide.jsonl \
        --rfl results/fo_rfl_wide.jsonl \
        --window 100

    # With default paths:
    uv run python experiments/generate_dyno_chart.py --window 100
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.plotting import make_dyno_chart, save_figure
import json
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Generate Dyno Chart: Baseline vs RFL abstention dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--baseline',
        type=str,
        default='results/fo_baseline_wide.jsonl',
        help='Path to baseline JSONL file (default: results/fo_baseline_wide.jsonl)'
    )
    
    parser.add_argument(
        '--rfl',
        type=str,
        default='results/fo_rfl_wide.jsonl',
        help='Path to RFL JSONL file (default: results/fo_rfl_wide.jsonl)'
    )
    
    parser.add_argument(
        '--window',
        type=int,
        default=100,
        help='Rolling window size for computing rolling mean (default: 100)'
    )
    
    parser.add_argument(
        '--output-name',
        type=str,
        default='rfl_dyno_chart',
        help='Output filename (without extension, default: rfl_dyno_chart)'
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    baseline_path = Path(args.baseline)
    rfl_path = Path(args.rfl)
    
    if not baseline_path.exists():
        print(f"Error: Baseline file not found: {baseline_path}", file=sys.stderr)
        sys.exit(1)
    
    if not rfl_path.exists():
        print(f"Error: RFL file not found: {rfl_path}", file=sys.stderr)
        sys.exit(1)
    
    # Generate the chart
    print(f"Loading baseline data from: {baseline_path}")
    print(f"Loading RFL data from: {rfl_path}")
    print(f"Using rolling window: {args.window}")
    
    try:
        # Generate chart with metadata
        fig, metadata = make_dyno_chart(
            baseline_path=str(baseline_path),
            rfl_path=str(rfl_path),
            window=args.window,
            return_metadata=True
        )
        
        # Save the figure
        saved_path = save_figure(args.output_name, fig)
        
        # Generate provenance manifest
        manifest = {
            'chart_name': args.output_name,
            'generated_at': datetime.now().isoformat(),
            'baseline_path': str(baseline_path.resolve()),
            'rfl_path': str(rfl_path.resolve()),
            'baseline_cycles': metadata['baseline_cycles'],
            'rfl_cycles': metadata['rfl_cycles'],
            'baseline_cycle_range': metadata['baseline_cycle_range'],
            'rfl_cycle_range': metadata['rfl_cycle_range'],
            'window_size': metadata['window_size'],
            'slice_name': metadata['slice_name'],
            'slice_display': metadata['slice_display'],
            'output_files': {
                'png': str(Path(saved_path).resolve()),
                'pdf': str(Path('artifacts/figures') / f"{args.output_name}.pdf")
            }
        }
        
        # Save manifest
        manifest_path = Path('artifacts/figures') / f"{args.output_name}_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✓ Dyno Chart saved to artifacts/figures/{args.output_name}.png")
        print(f"✓ Dyno Chart saved to artifacts/figures/{args.output_name}.pdf (if PDF backend available)")
        print(f"✓ Provenance manifest saved to artifacts/figures/{args.output_name}_manifest.json")
        print(f"\nChart Details:")
        print(f"  Baseline: {metadata['baseline_cycles']} cycles ({metadata['baseline_cycle_range']})")
        print(f"  RFL: {metadata['rfl_cycles']} cycles ({metadata['rfl_cycle_range']})")
        print(f"  Slice: {metadata['slice_display']}")
        print(f"  Window: {metadata['window_size']}")
        print(f"\nFull path: {Path(saved_path).resolve()}")
        
    except Exception as e:
        print(f"Error generating Dyno Chart: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

