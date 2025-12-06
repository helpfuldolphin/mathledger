#!/usr/bin/env python3
"""
Generate Triple-Hash Verification with SHA256 seal for MathLedger Composite Sprint.

Produces:
- artifacts/ci/triple_hash_summary.json
- artifacts/ci/triple_hash_verification.txt (with SHA256 seal)
- Composite attestation from DA-UI + DA-Reasoning + DA-Composite

Exit codes: 0=PASS, 1=FAIL
"""

import json
import hashlib
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


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


def check_onepagers() -> Dict[str, bool]:
    """Check if onepager PDFs exist."""
    return {
        'fol': os.path.exists('docs/onepager_fol.pdf'),
        'pl2': os.path.exists('docs/onepager_pl2.pdf')
    }


def verify_hash_integrity(merkle_roots: List[str]) -> Dict[str, Any]:
    """Verify hash integrity (disjoint baseline/guided merkle roots)."""
    if len(merkle_roots) < 6:
        return {
            'status': 'FAIL',
            'message': f'Insufficient merkle roots (found {len(merkle_roots)}, expected 6)',
            'unique_count': len(set(merkle_roots)),
            'total_count': len(merkle_roots),
            'baseline_roots': [],
            'guided_roots': []
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
            'overlap': overlap,
            'baseline_roots': baseline_roots,
            'guided_roots': guided_roots
        }
    else:
        return {
            'status': 'FAIL',
            'message': f'Hash integrity violation: baseline={baseline_unique}/3, guided={guided_unique}/3, overlap={overlap}',
            'baseline_unique': baseline_unique,
            'guided_unique': guided_unique,
            'overlap': overlap,
            'baseline_roots': baseline_roots,
            'guided_roots': guided_roots
        }


def compute_composite_attestation(ut_data: Dict, rt_data: Dict, ht_data: Dict) -> str:
    """Compute composite SHA256 attestation from U_t, R_t, H_t."""
    ut_str = f"uplift:{ut_data['uplift_ratio']:.6f}|baseline:{ut_data['baseline_mean']:.6f}|guided:{ut_data['guided_mean']:.6f}|p:{ut_data['p_value']:.10f}"
    rt_str = f"fol_onepager:{rt_data['onepager_fol']}|pl2_onepager:{rt_data['onepager_pl2']}"
    ht_str = f"baseline_unique:{ht_data['baseline_unique']}|guided_unique:{ht_data['guided_unique']}|overlap:{ht_data['overlap']}"
    
    composite_input = f"{ut_str}||{rt_str}||{ht_str}"
    return hashlib.sha256(composite_input.encode('utf-8')).hexdigest()


def load_da_attestations() -> Dict[str, Optional[str]]:
    """Load dual attestation merkle roots from workflow artifacts."""
    attestations = {
        'da_ui': None,
        'da_reasoning': None,
        'da_composite': None
    }
    
    da_paths = {
        'da_ui': 'artifacts/ci/da_ui_merkle.txt',
        'da_reasoning': 'artifacts/ci/da_reasoning_merkle.txt',
        'da_composite': 'artifacts/ci/da_composite_merkle.txt'
    }
    
    for key, path in da_paths.items():
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    attestations[key] = f.read().strip()
            except Exception:
                pass
    
    return attestations


def generate_triple_hash_summary() -> Dict[str, Any]:
    """Generate complete triple-hash summary with attestations."""
    fol_stats = load_fol_stats()
    onepagers = check_onepagers()
    merkle_roots = load_fol_csv()
    hash_integrity = verify_hash_integrity(merkle_roots)
    
    uplift_ratio = fol_stats.get('uplift_x', 0.0)
    p_value = fol_stats.get('p_value', 1.0)
    uplift_status = 'PASS' if uplift_ratio >= 1.30 and p_value < 0.05 else 'FAIL'
    
    ut_data = {
        'status': uplift_status,
        'uplift_ratio': uplift_ratio,
        'p_value': p_value,
        'baseline_mean': fol_stats.get('mean_baseline', 0.0),
        'guided_mean': fol_stats.get('mean_guided', 0.0),
        'threshold': '>=1.30x, p<0.05'
    }
    
    reproducibility_status = 'PASS' if onepagers['fol'] and onepagers['pl2'] else 'FAIL'
    
    rt_data = {
        'status': reproducibility_status,
        'onepager_fol': onepagers['fol'],
        'onepager_pl2': onepagers['pl2'],
        'message': 'Onepagers verified' if reproducibility_status == 'PASS' else 'Missing onepagers'
    }
    
    ht_data = {
        'status': hash_integrity['status'],
        'message': hash_integrity['message'],
        'baseline_unique': hash_integrity.get('baseline_unique', 0),
        'guided_unique': hash_integrity.get('guided_unique', 0),
        'overlap': hash_integrity.get('overlap', 0),
        'baseline_roots': hash_integrity.get('baseline_roots', []),
        'guided_roots': hash_integrity.get('guided_roots', [])
    }
    
    overall_status = 'PASS' if uplift_status == 'PASS' and reproducibility_status == 'PASS' and hash_integrity['status'] == 'PASS' else 'FAIL'
    
    composite_attestation = compute_composite_attestation(ut_data, rt_data, ht_data)
    
    da_attestations = load_da_attestations()
    
    return {
        'U_t': ut_data,
        'R_t': rt_data,
        'H_t': ht_data,
        'overall_status': overall_status,
        'composite_attestation': composite_attestation,
        'da_attestations': da_attestations,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'sprint_duration_hours': 72,
        'conductor': 'Devin J'
    }


def write_verification_file(summary: Dict[str, Any], output_path: str):
    """Write human-readable verification file with SHA256 seal."""
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MathLedger Composite Sprint - Triple-Hash Verification\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write(f"Conductor: {summary['conductor']}\n")
        f.write(f"Sprint Duration: {summary['sprint_duration_hours']} hours\n\n")
        
        ut = summary['U_t']
        f.write(f"[U_t] Uplift Verification: {ut['status']}\n")
        f.write(f"  Uplift Ratio:     {ut['uplift_ratio']:.2f}x\n")
        f.write(f"  Baseline Mean:    {ut['baseline_mean']:.2f} proofs/hour\n")
        f.write(f"  Guided Mean:      {ut['guided_mean']:.2f} proofs/hour\n")
        f.write(f"  P-value:          {ut['p_value']:.10f}\n")
        f.write(f"  Threshold:        {ut['threshold']}\n\n")
        
        rt = summary['R_t']
        f.write(f"[R_t] Reproducibility Verification: {rt['status']}\n")
        f.write(f"  FOL Onepager:     {'VERIFIED' if rt['onepager_fol'] else 'MISSING'}\n")
        f.write(f"  PL-2 Onepager:    {'VERIFIED' if rt['onepager_pl2'] else 'MISSING'}\n")
        f.write(f"  Message:          {rt['message']}\n\n")
        
        ht = summary['H_t']
        f.write(f"[H_t] Hash Integrity Verification: {ht['status']}\n")
        f.write(f"  Baseline Unique:  {ht['baseline_unique']}/3\n")
        f.write(f"  Guided Unique:    {ht['guided_unique']}/3\n")
        f.write(f"  Overlap:          {ht['overlap']}\n")
        f.write(f"  Message:          {ht['message']}\n\n")
        
        if ht['baseline_roots']:
            f.write("  Baseline Merkle Roots:\n")
            for i, root in enumerate(ht['baseline_roots'], 1):
                f.write(f"    seed{100+i}: {root}\n")
            f.write("\n")
        
        if ht['guided_roots']:
            f.write("  Guided Merkle Roots:\n")
            for i, root in enumerate(ht['guided_roots'], 1):
                f.write(f"    seed{100+i}: {root}\n")
            f.write("\n")
        
        da = summary['da_attestations']
        f.write("Dual Attestation Status:\n")
        f.write(f"  DA-UI:        {da['da_ui'] or 'PENDING'}\n")
        f.write(f"  DA-Reasoning: {da['da_reasoning'] or 'PENDING'}\n")
        f.write(f"  DA-Composite: {da['da_composite'] or 'PENDING'}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write(f"OVERALL STATUS: {summary['overall_status']}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Composite Attestation (SHA256):\n")
        f.write(f"{summary['composite_attestation']}\n\n")
        
        f.write("Verification Formula:\n")
        f.write("  SHA256(U_t || R_t || H_t)\n")
        f.write("  where:\n")
        f.write("    U_t = uplift_ratio + baseline_mean + guided_mean + p_value\n")
        f.write("    R_t = fol_onepager + pl2_onepager\n")
        f.write("    H_t = baseline_unique + guided_unique + overlap\n\n")
        
        f.write("=" * 70 + "\n")
        f.write(f"[{summary['overall_status']}] Triple-Hash Verification <{summary['composite_attestation']}>\n")
        f.write("=" * 70 + "\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate Triple-Hash Verification for MathLedger Composite Sprint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts/ci',
        help='Output directory for artifacts (default: artifacts/ci)'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    summary = generate_triple_hash_summary()
    
    json_path = os.path.join(args.output_dir, 'triple_hash_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Generated: {json_path}")
    
    txt_path = os.path.join(args.output_dir, 'triple_hash_verification.txt')
    write_verification_file(summary, txt_path)
    print(f"Generated: {txt_path}")
    
    print("\n" + "=" * 70)
    print(f"[{summary['overall_status']}] Triple-Hash Verification")
    print("=" * 70)
    print(f"U_t (Uplift):         {summary['U_t']['uplift_ratio']:.2f}x {summary['U_t']['status']}")
    print(f"R_t (Reproducibility): {summary['R_t']['status']}")
    print(f"H_t (Hash Integrity):  {summary['H_t']['status']}")
    print(f"Composite Attestation: {summary['composite_attestation']}")
    print("=" * 70)
    
    sys.exit(0 if summary['overall_status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
