#!/usr/bin/env python3
"""
Metrics Diff - Compare metrics across derivation runs

Compares run_metrics.jsonl entries to identify performance changes,
success rate variations, and configuration differences.

Usage:
    python metrics_diff.py --baseline run1.jsonl --current run2.jsonl
    python metrics_diff.py --seed 42 --seed 43
    python metrics_diff.py --last 10  # Compare last 10 runs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).parent.parent.parent
METRICS_FILE = REPO_ROOT / 'artifacts' / 'wpv5' / 'run_metrics.jsonl'

def load_metrics(file_path: Path) -> List[Dict[str, Any]]:
    """Load metrics from JSONL file"""
    metrics = []
    if not file_path.exists():
        return metrics
    
    with open(file_path) as f:
        for line in f:
            if line.strip():
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return metrics

def find_by_seed(metrics: List[Dict], seed: int) -> Dict:
    """Find metric entry by seed"""
    for entry in metrics:
        if entry.get('seed') == seed:
            return entry
    return {}

def compute_diff(baseline: Dict, current: Dict) -> Dict:
    """Compute differences between two metric entries"""
    diff = {
        'baseline': {},
        'current': {},
        'changes': {}
    }
    
    keys = ['system', 'mode', 'method', 'seed', 'inserted_proofs', 
            'wall_minutes', 'block_no', 'merkle']
    
    for key in keys:
        b_val = baseline.get(key)
        c_val = current.get(key)
        
        if b_val != c_val:
            diff['baseline'][key] = b_val
            diff['current'][key] = c_val
            
            if isinstance(b_val, (int, float)) and isinstance(c_val, (int, float)):
                delta = c_val - b_val
                pct = (delta / b_val * 100) if b_val != 0 else 0
                diff['changes'][key] = {
                    'delta': delta,
                    'percent': round(pct, 2)
                }
    
    return diff

def format_diff(diff: Dict) -> str:
    """Format diff for ASCII output"""
    lines = []
    lines.append("=== METRICS DIFF ===")
    lines.append("")
    
    if not diff['changes']:
        lines.append("No differences found")
        return '\n'.join(lines)
    
    lines.append("BASELINE:")
    for key, val in diff['baseline'].items():
        lines.append(f"  {key}: {val}")
    
    lines.append("")
    lines.append("CURRENT:")
    for key, val in diff['current'].items():
        lines.append(f"  {key}: {val}")
    
    lines.append("")
    lines.append("CHANGES:")
    for key, change in diff['changes'].items():
        if 'delta' in change:
            sign = '+' if change['delta'] > 0 else ''
            lines.append(f"  {key}: {sign}{change['delta']} ({sign}{change['percent']}%)")
        else:
            lines.append(f"  {key}: changed")
    
    return '\n'.join(lines)

def compare_last_n(metrics: List[Dict], n: int) -> str:
    """Compare last N runs"""
    if len(metrics) < 2:
        return "Not enough metrics to compare"
    
    recent = metrics[-n:] if len(metrics) >= n else metrics
    
    lines = []
    lines.append(f"=== LAST {len(recent)} RUNS ===")
    lines.append("")
    
    total_proofs = sum(m.get('inserted_proofs', 0) for m in recent)
    avg_proofs = total_proofs / len(recent)
    total_time = sum(m.get('wall_minutes', 0) for m in recent)
    avg_time = total_time / len(recent)
    
    lines.append(f"Total runs: {len(recent)}")
    lines.append(f"Total proofs: {total_proofs}")
    lines.append(f"Average proofs/run: {avg_proofs:.1f}")
    lines.append(f"Average time: {avg_time:.1f} minutes")
    lines.append("")
    
    lines.append("RUN BREAKDOWN:")
    for i, m in enumerate(recent, 1):
        seed = m.get('seed', 'N/A')
        proofs = m.get('inserted_proofs', 0)
        time = m.get('wall_minutes', 0)
        method = m.get('method', 'unknown')
        lines.append(f"  {i}. seed={seed} proofs={proofs} time={time:.1f}m method={method}")
    
    return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(description='Compare metrics across runs')
    parser.add_argument('--baseline', type=Path, help='Baseline metrics file')
    parser.add_argument('--current', type=Path, help='Current metrics file')
    parser.add_argument('--seed', type=int, nargs=2, metavar=('BASELINE', 'CURRENT'),
                       help='Compare two runs by seed')
    parser.add_argument('--last', type=int, help='Compare last N runs')
    parser.add_argument('--file', type=Path, default=METRICS_FILE,
                       help='Metrics file to use (default: artifacts/wpv5/run_metrics.jsonl)')
    
    args = parser.parse_args()
    
    if args.baseline and args.current:
        baseline_metrics = load_metrics(args.baseline)
        current_metrics = load_metrics(args.current)
        
        if not baseline_metrics or not current_metrics:
            print("Error: Could not load metrics from files")
            return 1
        
        diff = compute_diff(baseline_metrics[0], current_metrics[0])
        print(format_diff(diff))
    
    elif args.seed:
        metrics = load_metrics(args.file)
        baseline = find_by_seed(metrics, args.seed[0])
        current = find_by_seed(metrics, args.seed[1])
        
        if not baseline or not current:
            print(f"Error: Could not find seeds {args.seed[0]} or {args.seed[1]}")
            return 1
        
        diff = compute_diff(baseline, current)
        print(format_diff(diff))
    
    elif args.last:
        metrics = load_metrics(args.file)
        print(compare_last_n(metrics, args.last))
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
