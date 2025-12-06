#!/usr/bin/env python
"""Detailed analysis of uplift experiment results."""
import json
import sys
from collections import Counter
from typing import Dict, List, Any
import statistics

def analyze_file(path: str) -> Dict[str, Any]:
    """Analyze a single JSONL file in detail."""
    data: List[Dict[str, Any]] = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    
    if not data:
        return {}
    
    # Basic stats
    total = len(data)
    successes = sum(1 for d in data if d.get('success', False))
    verified_counts = [d.get('derivation', {}).get('verified', 0) for d in data]
    
    # Distribution of verified counts
    verified_dist = Counter(verified_counts)
    
    # Policy weights (if present)
    policy_weights = []
    for d in data:
        pw = d.get('policy_weights')
        if pw:
            policy_weights.append(pw)
    
    # Success by verified count
    success_by_verified = {}
    for d in data:
        verified = d.get('derivation', {}).get('verified', 0)
        success = d.get('success', False)
        if verified not in success_by_verified:
            success_by_verified[verified] = {'total': 0, 'success': 0}
        success_by_verified[verified]['total'] += 1
        if success:
            success_by_verified[verified]['success'] += 1
    
    return {
        'total': total,
        'successes': successes,
        'success_rate': successes / total if total > 0 else 0.0,
        'avg_verified': statistics.mean(verified_counts) if verified_counts else 0.0,
        'median_verified': statistics.median(verified_counts) if verified_counts else 0.0,
        'std_verified': statistics.stdev(verified_counts) if len(verified_counts) > 1 else 0.0,
        'verified_distribution': dict(verified_dist),
        'success_by_verified': success_by_verified,
        'policy_weights': policy_weights,
        'has_policy': len(policy_weights) > 0,
    }

def main(baseline_path: str, rfl_path: str) -> None:
    """Compare baseline and RFL results."""
    baseline = analyze_file(baseline_path)
    rfl = analyze_file(rfl_path)
    
    print("=" * 70)
    print("DETAILED UPLIFT ANALYSIS")
    print("=" * 70)
    print()
    
    # Basic comparison
    print("BASIC STATISTICS:")
    print(f"  Baseline: {baseline['successes']}/{baseline['total']} = {baseline['success_rate']:.3f}")
    print(f"  RFL:      {rfl['successes']}/{rfl['total']} = {rfl['success_rate']:.3f}")
    print(f"  Δ:        {rfl['success_rate'] - baseline['success_rate']:+.3f}")
    print()
    
    # Verified count statistics
    print("VERIFIED PROOFS PER CYCLE:")
    print(f"  Baseline: mean={baseline['avg_verified']:.2f}, median={baseline['median_verified']:.1f}, std={baseline['std_verified']:.2f}")
    print(f"  RFL:      mean={rfl['avg_verified']:.2f}, median={rfl['median_verified']:.1f}, std={rfl['std_verified']:.2f}")
    print()
    
    # Distribution comparison
    print("DISTRIBUTION OF VERIFIED COUNTS:")
    all_verified = set(baseline['verified_distribution'].keys()) | set(rfl['verified_distribution'].keys())
    for v in sorted(all_verified):
        base_count = baseline['verified_distribution'].get(v, 0)
        rfl_count = rfl['verified_distribution'].get(v, 0)
        base_pct = (base_count / baseline['total']) * 100 if baseline['total'] > 0 else 0
        rfl_pct = (rfl_count / rfl['total']) * 100 if rfl['total'] > 0 else 0
        print(f"  {v} verified: Baseline={base_count:3d} ({base_pct:5.1f}%) | RFL={rfl_count:3d} ({rfl_pct:5.1f}%) | Δ={rfl_count-base_count:+3d}")
    print()
    
    # Success rate by verified count
    print("SUCCESS RATE BY VERIFIED COUNT:")
    all_verified_success = set(baseline['success_by_verified'].keys()) | set(rfl['success_by_verified'].keys())
    for v in sorted(all_verified_success):
        base = baseline['success_by_verified'].get(v, {'total': 0, 'success': 0})
        rfl_data = rfl['success_by_verified'].get(v, {'total': 0, 'success': 0})
        base_rate = (base['success'] / base['total']) * 100 if base['total'] > 0 else 0
        rfl_rate = (rfl_data['success'] / rfl_data['total']) * 100 if rfl_data['total'] > 0 else 0
        print(f"  {v} verified: Baseline={base_rate:5.1f}% ({base['success']}/{base['total']}) | RFL={rfl_rate:5.1f}% ({rfl_data['success']}/{rfl_data['total']})")
    print()
    
    # Policy weight analysis
    if rfl['has_policy']:
        print("RFL POLICY WEIGHTS:")
        if rfl['policy_weights']:
            first = rfl['policy_weights'][0]
            last = rfl['policy_weights'][-1]
            print(f"  Initial: len={first.get('len', 0):.4f}, depth={first.get('depth', 0):.4f}")
            print(f"  Final:   len={last.get('len', 0):.4f}, depth={last.get('depth', 0):.4f}")
            print(f"  Change:  len={last.get('len', 0) - first.get('len', 0):+.4f}, depth={last.get('depth', 0) - first.get('depth', 0):+.4f}")
            
            # Check if weights changed significantly
            len_change = abs(last.get('len', 0) - first.get('len', 0))
            depth_change = abs(last.get('depth', 0) - first.get('depth', 0))
            if len_change > 0.01 or depth_change > 0.01:
                print(f"  → Policy weights changed significantly (len: {len_change:.4f}, depth: {depth_change:.4f})")
            else:
                print(f"  → Policy weights changed minimally")
        print()
    
    # Interpretation
    print("INTERPRETATION:")
    delta = rfl['success_rate'] - baseline['success_rate']
    if abs(delta) < 0.05:
        print("  Case A: RFL ≈ Baseline (no clear uplift)")
        print("  Possible reasons:")
        print("    - Policy weights changed but didn't affect candidate ordering enough")
        print("    - Features (length/depth) not discriminative for this slice")
        print("    - Search space too easy - even different ordering finds same proofs")
        if rfl['has_policy'] and rfl['policy_weights']:
            len_change = abs(rfl['policy_weights'][-1].get('len', 0) - rfl['policy_weights'][0].get('len', 0))
            if len_change > 0.1:
                print(f"    - Policy DID update significantly (weights changed by {len_change:.3f})")
                print("      but search behavior didn't change → need better features or stronger coupling")
    elif delta > 0.05:
        print(f"  Case C: RFL BETTER than baseline (uplift = {delta:.3f})")
    elif delta < -0.05:
        print(f"  Case B: RFL WORSE than baseline (negative uplift = {delta:.3f})")
        print("  This suggests policy is learning in wrong direction - may need sign flip")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python detailed_uplift_analysis.py <baseline.jsonl> <rfl.jsonl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

