#!/usr/bin/env python3
"""
Uplift Gate for FOL and PL-2 Systems with Dual-Attestation Artifacts.

Processes A/B CSV splits, calculates uplift ratios, generates badges,
and creates summary reports for CI integration with verified metric hashes.

Exit codes: 0=PASS, 1=FAIL, 2=ERROR, 3=WARNING (variance)
"""

import argparse
import csv
import json
import os
import sys
import hashlib
import statistics as stats
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from backend.ledger.blockchain import merkle_root
except ImportError:
    def merkle_root(ids: List[str]) -> str:
        """Fallback merkle root implementation using SHA256."""
        if not ids:
            return hashlib.sha256(b"").hexdigest()
        
        sorted_ids = sorted(ids)
        combined = "|".join(sorted_ids)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def load_csv_data(csv_path: str) -> List[Dict[str, Any]]:
    """Load CSV data with error handling."""
    if not os.path.exists(csv_path):
        return []
    
    rows = []
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}", file=sys.stderr)
        return []
    
    return rows


def extract_performance_data(rows: List[Dict[str, Any]], system: str) -> Tuple[List[float], List[float]]:
    """Extract baseline and guided performance data for a system."""
    baseline_pph = []
    guided_pph = []
    
    for row in rows:
        mode = row.get('mode', '').lower()
        pph_str = row.get('proofs_per_hour', '')
        
        try:
            pph = float(pph_str)
            if system.lower() in mode:
                if 'baseline' in mode:
                    baseline_pph.append(pph)
                elif 'guided' in mode:
                    guided_pph.append(pph)
        except (ValueError, TypeError):
            continue
    
    return baseline_pph, guided_pph


def calculate_uplift(baseline: List[float], guided: List[float]) -> Dict[str, Any]:
    """Calculate uplift metrics with statistical analysis and regression-floor rules."""
    if not baseline or not guided:
        return {
            'uplift_ratio': 0.0,
            'baseline_mean': 0.0,
            'guided_mean': 0.0,
            'status': 'ERROR',
            'message': 'Insufficient data'
        }
    
    baseline_mean = stats.mean(baseline)
    guided_mean = stats.mean(guided)
    uplift_ratio = guided_mean / max(baseline_mean, 1e-9)
    
    if uplift_ratio >= 1.25:
        status = 'PASS'
        message = f'Uplift {uplift_ratio:.2f}x >= 1.25x threshold'
    elif uplift_ratio >= 1.0:
        status = 'WARNING'
        message = f'Uplift {uplift_ratio:.2f}x below 1.25x threshold but positive'
    else:
        status = 'FAIL'
        message = f'Uplift {uplift_ratio:.2f}x indicates regression (< 1.0x floor)'
    
    return {
        'uplift_ratio': uplift_ratio,
        'baseline_mean': baseline_mean,
        'guided_mean': guided_mean,
        'baseline_samples': len(baseline),
        'guided_samples': len(guided),
        'status': status,
        'message': message
    }


def generate_badge_json(label: str, message: str, color: str) -> Dict[str, Any]:
    """Generate shields.io compatible badge JSON."""
    return {
        'schemaVersion': 1,
        'label': label,
        'message': message,
        'color': color
    }


def generate_merkle_hashes(fol_metrics: Dict[str, Any], pl2_metrics: Dict[str, Any], 
                            performance_passport_path: str = "performance_passport.json") -> Dict[str, str]:
    """Generate merkle_perf and merkle_export hashes for dual-attestation."""
    hashes = {}
    
    perf_data = [
        f"fol_uplift:{fol_metrics['uplift_ratio']:.6f}",
        f"fol_baseline:{fol_metrics['baseline_mean']:.6f}",
        f"fol_guided:{fol_metrics['guided_mean']:.6f}",
        f"pl2_uplift:{pl2_metrics['uplift_ratio']:.6f}",
        f"pl2_baseline:{pl2_metrics['baseline_mean']:.6f}",
        f"pl2_guided:{pl2_metrics['guided_mean']:.6f}"
    ]
    hashes['merkle_perf'] = merkle_root(perf_data)
    
    if os.path.exists(performance_passport_path):
        try:
            with open(performance_passport_path, 'r') as f:
                passport_data = json.load(f)
            export_data = [
                f"cartographer:{passport_data.get('cartographer', 'unknown')}",
                f"run_id:{passport_data.get('run_id', 'unknown')}",
                f"overall_status:{passport_data.get('summary', {}).get('overall_status', 'unknown')}"
            ]
            hashes['merkle_export'] = merkle_root(export_data)
        except Exception:
            hashes['merkle_export'] = hashlib.sha256(b"passport_unavailable").hexdigest()
    else:
        hashes['merkle_export'] = hashlib.sha256(b"passport_not_found").hexdigest()
    
    return hashes


def generate_badges(fol_metrics: Dict[str, Any], pl2_metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Generate all badge JSONs."""
    badges = {}
    
    fol_color = 'green' if fol_metrics['status'] == 'PASS' else 'yellow' if fol_metrics['status'] == 'WARNING' else 'red'
    badges['fol_perf_badge'] = generate_badge_json(
        'FOL Uplift',
        f"{fol_metrics['uplift_ratio']:.2f}x",
        fol_color
    )
    
    pl2_color = 'green' if pl2_metrics['status'] == 'PASS' else 'yellow' if pl2_metrics['status'] == 'WARNING' else 'red'
    badges['pl2_perf_badge'] = generate_badge_json(
        'PL-2 Uplift',
        f"{pl2_metrics['uplift_ratio']:.2f}x",
        pl2_color
    )
    
    overall_status = 'PASS' if fol_metrics['status'] == 'PASS' and pl2_metrics['status'] == 'PASS' else 'FAIL'
    overall_color = 'green' if overall_status == 'PASS' else 'red'
    avg_uplift = (fol_metrics['uplift_ratio'] + pl2_metrics['uplift_ratio']) / 2
    
    badges['uplift_badge'] = generate_badge_json(
        'Dual Uplift',
        f"{avg_uplift:.2f}x avg",
        overall_color
    )
    
    return badges


def generate_verified_badges(fol_metrics: Dict[str, Any], pl2_metrics: Dict[str, Any], 
                           merkle_hashes: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Generate verified badges with dual-attestation."""
    badges = generate_badges(fol_metrics, pl2_metrics)
    
    overall_status = 'PASS' if (fol_metrics['status'] in ['PASS', 'WARNING'] and 
                               pl2_metrics['status'] in ['PASS', 'WARNING']) else 'FAIL'
    overall_color = 'green' if overall_status == 'PASS' else 'red'
    
    avg_uplift = (fol_metrics['uplift_ratio'] + pl2_metrics['uplift_ratio']) / 2
    badges['uplift_verified_badge'] = generate_badge_json(
        'Verified Uplift',
        f"{avg_uplift:.2f}x ✓",
        overall_color
    )
    
    return badges


def generate_summary_md(fol_metrics: Dict[str, Any], pl2_metrics: Dict[str, Any], timestamp: str) -> str:
    """Generate uplift summary markdown."""
    return f"""# Uplift Summary Report

Generated: {timestamp}

- **Baseline Mean**: {fol_metrics['baseline_mean']:.2f} proofs/hour
- **Guided Mean**: {fol_metrics['guided_mean']:.2f} proofs/hour
- **Uplift Ratio**: {fol_metrics['uplift_ratio']:.2f}x
- **Status**: {fol_metrics['status']}
- **Samples**: {fol_metrics.get('baseline_samples', 0)} baseline, {fol_metrics.get('guided_samples', 0)} guided

- **Baseline Mean**: {pl2_metrics['baseline_mean']:.2f} proofs/hour
- **Guided Mean**: {pl2_metrics['guided_mean']:.2f} proofs/hour
- **Uplift Ratio**: {pl2_metrics['uplift_ratio']:.2f}x
- **Status**: {pl2_metrics['status']}
- **Samples**: {pl2_metrics.get('baseline_samples', 0)} baseline, {pl2_metrics.get('guided_samples', 0)} guided

- **Threshold**: 1.25x minimum uplift required
- **FOL Result**: {fol_metrics['message']}
- **PL-2 Result**: {pl2_metrics['message']}
"""


def generate_verified_summary_md(fol_metrics: Dict[str, Any], pl2_metrics: Dict[str, Any], 
                               merkle_hashes: Dict[str, str], timestamp: str) -> str:
    """Generate verified uplift summary markdown with dual-attestation."""
    return f"""# Verified Uplift Summary Report

Generated: {timestamp}

- **Baseline Mean**: {fol_metrics['baseline_mean']:.2f} proofs/hour
- **Guided Mean**: {fol_metrics['guided_mean']:.2f} proofs/hour
- **Uplift Ratio**: {fol_metrics['uplift_ratio']:.2f}x
- **Status**: {fol_metrics['status']}
- **Samples**: {fol_metrics.get('baseline_samples', 0)} baseline, {fol_metrics.get('guided_samples', 0)} guided

- **Baseline Mean**: {pl2_metrics['baseline_mean']:.2f} proofs/hour
- **Guided Mean**: {pl2_metrics['guided_mean']:.2f} proofs/hour
- **Uplift Ratio**: {pl2_metrics['uplift_ratio']:.2f}x
- **Status**: {pl2_metrics['status']}
- **Samples**: {pl2_metrics.get('baseline_samples', 0)} baseline, {pl2_metrics.get('guided_samples', 0)} guided

- **Merkle Performance Hash**: {merkle_hashes['merkle_perf']}
- **Merkle Export Hash**: {merkle_hashes['merkle_export']}
- **Regression Floor**: 1.0x minimum (< 1.0 → FAIL, 1.0-1.25 → WARN, ≥ 1.25 → PASS)
- **FOL Result**: {fol_metrics['message']}
- **PL-2 Result**: {pl2_metrics['message']}
"""


def main():
    """Main entry point for uplift gate processing with dual-attestation."""
    parser = argparse.ArgumentParser(
        description='Uplift Gate for FOL and PL-2 Systems with Dual-Attestation'
    )
    parser.add_argument('--fol-csv', default='artifacts/wpv5/fol_ab.csv',
                       help='Path to FOL A/B CSV data')
    parser.add_argument('--pl2-csv', default='artifacts/wpv5/pl2_ab.csv',
                       help='Path to PL-2 A/B CSV data')
    parser.add_argument('--output-dir', default='artifacts/badges',
                       help='Output directory for badges and summary')
    parser.add_argument('--passport-path', default='performance_passport.json',
                       help='Path to performance passport JSON file')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    fol_data = load_csv_data(args.fol_csv)
    pl2_data = load_csv_data(args.pl2_csv)
    
    fol_baseline, fol_guided = extract_performance_data(fol_data, 'fol')
    pl2_baseline, pl2_guided = extract_performance_data(pl2_data, 'pl2')
    
    fol_metrics = calculate_uplift(fol_baseline, fol_guided)
    pl2_metrics = calculate_uplift(pl2_baseline, pl2_guided)
    
    merkle_hashes = generate_merkle_hashes(fol_metrics, pl2_metrics, args.passport_path)
    
    badges = generate_verified_badges(fol_metrics, pl2_metrics, merkle_hashes)
    
    for badge_name, badge_data in badges.items():
        badge_path = os.path.join(args.output_dir, f'{badge_name}.json')
        with open(badge_path, 'w') as f:
            json.dump(badge_data, f, indent=2)
    
    timestamp = datetime.now().isoformat()
    summary_md = generate_verified_summary_md(fol_metrics, pl2_metrics, merkle_hashes, timestamp)
    summary_path = os.path.join(args.output_dir, 'uplift_verified_summary.md')
    with open(summary_path, 'w') as f:
        f.write(summary_md)
    
    summary_json = {
        'fol_metrics': fol_metrics,
        'pl2_metrics': pl2_metrics,
        'merkle_hashes': merkle_hashes,
        'timestamp': timestamp,
        'overall_status': 'PASS' if (fol_metrics['status'] in ['PASS', 'WARNING'] and 
                                   pl2_metrics['status'] in ['PASS', 'WARNING']) else 'FAIL'
    }
    summary_json_path = os.path.join(args.output_dir, 'uplift_verified_summary.json')
    with open(summary_json_path, 'w') as f:
        json.dump(summary_json, f, indent=2)
    
    print(f"FOL Uplift: {fol_metrics['uplift_ratio']:.2f}x ({fol_metrics['status']})")
    print(f"PL-2 Uplift: {pl2_metrics['uplift_ratio']:.2f}x ({pl2_metrics['status']})")
    print(f"Merkle Performance Hash: {merkle_hashes['merkle_perf']}")
    print(f"Merkle Export Hash: {merkle_hashes['merkle_export']}")
    
    if fol_metrics['status'] == 'ERROR' or pl2_metrics['status'] == 'ERROR':
        sys.exit(2)  # ERROR
    elif fol_metrics['status'] == 'FAIL' or pl2_metrics['status'] == 'FAIL':
        sys.exit(1)  # FAIL
    elif fol_metrics['status'] == 'WARNING' or pl2_metrics['status'] == 'WARNING':
        sys.exit(3)  # WARNING
    else:
        sys.exit(0)  # PASS


if __name__ == '__main__':
    main()
