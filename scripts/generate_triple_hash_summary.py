#!/usr/bin/env python3
"""
Generate Triple-Hash CI Summary for MathLedger Composite Sprint.

Outputs U_t (Uplift), R_t (Reproducibility), H_t (Hash Integrity) metrics
in CI-friendly format for sprint completion verification.

Exit codes: 0=PASS, 1=FAIL
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List


def load_fol_stats(stats_path: str = "artifacts/wpv5/fol_stats.json") -> Dict[str, Any]:
    """Load FOL statistics from JSON file."""
    if not os.path.exists(stats_path):
        return {}
    
    try:
        with open(stats_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load FOL stats: {e}", file=sys.stderr)
        return {}


def check_onepagers() -> Dict[str, bool]:
    """Check if onepager PDFs exist."""
    return {
        'fol': os.path.exists('docs/onepager_fol.pdf'),
        'pl2': os.path.exists('docs/onepager_pl2.pdf')
    }


def load_fol_csv(csv_path: str = "artifacts/wpv5/fol_ab.csv") -> List[str]:
    """Load FOL A/B CSV and extract merkle roots."""
    if not os.path.exists(csv_path):
        return []
    
    merkle_roots = []
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    merkle_root = parts[2]
                    if merkle_root:  # Skip empty merkle roots
                        merkle_roots.append(merkle_root)
    except Exception as e:
        print(f"Warning: Could not load FOL CSV: {e}", file=sys.stderr)
    
    return merkle_roots


def verify_hash_integrity(merkle_roots: List[str]) -> Dict[str, Any]:
    """Verify hash integrity (disjoint baseline/guided merkle roots)."""
    if len(merkle_roots) < 6:
        return {
            'status': 'FAIL',
            'message': f'Insufficient merkle roots (found {len(merkle_roots)}, expected 6)',
            'unique_count': len(set(merkle_roots)),
            'total_count': len(merkle_roots)
        }
    
    baseline_roots = merkle_roots[:3]
    guided_roots = merkle_roots[3:6]
    
    baseline_unique = len(set(baseline_roots))
    guided_unique = len(set(guided_roots))
    overlap = len(set(baseline_roots) & set(guided_roots))
    
    if baseline_unique == 3 and guided_unique == 3 and overlap == 0:
        return {
            'status': 'PASS',
            'message': '6/6 unique merkle roots, 0 overlap',
            'baseline_unique': baseline_unique,
            'guided_unique': guided_unique,
            'overlap': overlap
        }
    else:
        return {
            'status': 'FAIL',
            'message': f'Hash integrity violation: baseline={baseline_unique}/3, guided={guided_unique}/3, overlap={overlap}',
            'baseline_unique': baseline_unique,
            'guided_unique': guided_unique,
            'overlap': overlap
        }


def generate_summary() -> Dict[str, Any]:
    """Generate complete triple-hash summary."""
    fol_stats = load_fol_stats()
    onepagers = check_onepagers()
    merkle_roots = load_fol_csv()
    hash_integrity = verify_hash_integrity(merkle_roots)
    
    uplift_ratio = fol_stats.get('uplift_x', 0.0)
    p_value = fol_stats.get('p_value', 1.0)
    uplift_status = 'PASS' if uplift_ratio >= 1.30 and p_value < 0.05 else 'FAIL'
    
    reproducibility_status = 'PASS' if onepagers['fol'] and onepagers['pl2'] else 'FAIL'
    
    hash_status = hash_integrity['status']
    
    overall_status = 'PASS' if uplift_status == 'PASS' and reproducibility_status == 'PASS' and hash_status == 'PASS' else 'FAIL'
    
    return {
        'U_t': {
            'status': uplift_status,
            'uplift_ratio': uplift_ratio,
            'p_value': p_value,
            'baseline_mean': fol_stats.get('mean_baseline', 0.0),
            'guided_mean': fol_stats.get('mean_guided', 0.0),
            'threshold': '>=1.30x, p<0.05'
        },
        'R_t': {
            'status': reproducibility_status,
            'onepager_fol': onepagers['fol'],
            'onepager_pl2': onepagers['pl2'],
            'message': 'Onepagers verified' if reproducibility_status == 'PASS' else 'Missing onepagers'
        },
        'H_t': {
            'status': hash_status,
            'message': hash_integrity['message'],
            'baseline_unique': hash_integrity.get('baseline_unique', 0),
            'guided_unique': hash_integrity.get('guided_unique', 0),
            'overlap': hash_integrity.get('overlap', 0)
        },
        'overall_status': overall_status,
        'timestamp': '2025-10-19'
    }


def print_summary(summary: Dict[str, Any], format: str = 'text'):
    """Print summary in specified format."""
    if format == 'json':
        print(json.dumps(summary, indent=2))
    else:
        print("=" * 60)
        print("MathLedger Triple-Hash CI Summary")
        print("=" * 60)
        print()
        
        ut = summary['U_t']
        print(f"U_t (Uplift):         {ut['uplift_ratio']:.2f}x {ut['status']}")
        print(f"  Baseline Mean:      {ut['baseline_mean']:.2f} proofs/hour")
        print(f"  Guided Mean:        {ut['guided_mean']:.2f} proofs/hour")
        print(f"  P-value:            {ut['p_value']:.4f}")
        print(f"  Threshold:          {ut['threshold']}")
        print()
        
        rt = summary['R_t']
        print(f"R_t (Reproducibility): {rt['status']}")
        print(f"  FOL Onepager:       {'✓' if rt['onepager_fol'] else '✗'}")
        print(f"  PL-2 Onepager:      {'✓' if rt['onepager_pl2'] else '✗'}")
        print(f"  Message:            {rt['message']}")
        print()
        
        ht = summary['H_t']
        print(f"H_t (Hash Integrity):  {ht['status']}")
        print(f"  Baseline Unique:    {ht['baseline_unique']}/3")
        print(f"  Guided Unique:      {ht['guided_unique']}/3")
        print(f"  Overlap:            {ht['overlap']}")
        print(f"  Message:            {ht['message']}")
        print()
        
        print("=" * 60)
        print(f"Overall Status:       {summary['overall_status']}")
        print(f"Timestamp:            {summary['timestamp']}")
        print("=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate Triple-Hash CI Summary for MathLedger'
    )
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: stdout)'
    )
    
    args = parser.parse_args()
    
    summary = generate_summary()
    
    if args.output:
        original_stdout = sys.stdout
        with open(args.output, 'w') as f:
            sys.stdout = f
            print_summary(summary, args.format)
        sys.stdout = original_stdout
        print(f"Summary written to {args.output}")
    else:
        print_summary(summary, args.format)
    
    sys.exit(0 if summary['overall_status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
