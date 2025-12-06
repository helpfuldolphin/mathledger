#!/usr/bin/env python
"""Compare baseline vs RFL uplift experiment results."""
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

def analyze_file(path: str) -> Dict[str, Any]:
    """Analyze a single JSONL file."""
    data: List[Dict[str, Any]] = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    
    if not data:
        return {"total": 0, "success": 0, "success_rate": 0.0, "avg_verified": 0.0}
    
    successes = sum(1 for d in data if d.get('success', False))
    verified_counts = [d.get('derivation', {}).get('verified', 0) for d in data]
    
    # Check for RFL-specific fields
    rfl_cycles = [d for d in data if d.get('rfl', {}).get('executed', False)]
    policy_updates = []
    policy_weights_trajectory = []
    
    for d in data:
        rfl = d.get('rfl', {})
        if rfl.get('executed', False):
            policy_updates.append(rfl.get('policy_update', False))
            weights = d.get('policy_weights')
            if weights:
                policy_weights_trajectory.append(weights.copy())
    
    return {
        "total": len(data),
        "success": successes,
        "success_rate": successes / len(data) if data else 0.0,
        "avg_verified": sum(verified_counts) / len(verified_counts) if verified_counts else 0.0,
        "verified_counts": verified_counts[:10],  # First 10 for inspection
        "is_rfl": len(rfl_cycles) > 0,
        "policy_updates_count": sum(1 for p in policy_updates if p) if policy_updates else 0,
        "policy_weights_first": policy_weights_trajectory[0] if policy_weights_trajectory else None,
        "policy_weights_last": policy_weights_trajectory[-1] if policy_weights_trajectory else None,
    }

def main(baseline_path: str, rfl_path: str) -> None:
    """Compare baseline and RFL results."""
    baseline = analyze_file(baseline_path)
    rfl = analyze_file(rfl_path)
    
    print("=" * 60)
    print("UPLIFT EXPERIMENT COMPARISON")
    print("=" * 60)
    print()
    
    print("BASELINE:")
    print(f"  Cycles:        {baseline['total']}")
    print(f"  Successes:     {baseline['success']}")
    print(f"  Success rate:  {baseline['success_rate']:.3f}")
    print(f"  Avg verified:  {baseline['avg_verified']:.2f}")
    print()
    
    print("RFL:")
    print(f"  Cycles:        {rfl['total']}")
    print(f"  Successes:     {rfl['success']}")
    print(f"  Success rate:  {rfl['success_rate']:.3f}")
    print(f"  Avg verified:  {rfl['avg_verified']:.2f}")
    print()
    
    if rfl['is_rfl']:
        print("RFL POLICY UPDATES:")
        print(f"  Policy updates applied: {rfl['policy_updates_count']}/{rfl['total']}")
        if rfl['policy_weights_first']:
            print(f"  Initial weights:       {rfl['policy_weights_first']}")
        if rfl['policy_weights_last']:
            print(f"  Final weights:         {rfl['policy_weights_last']}")
        print()
    
    # Compute uplift
    delta_success = rfl['success_rate'] - baseline['success_rate']
    delta_absolute = rfl['success'] - baseline['success']
    
    print("UPLIFT:")
    print(f"  Δ success rate:  {delta_success:+.3f}")
    print(f"  Δ absolute:      {delta_absolute:+d} cycles")
    print()
    
    # Rough statistical check (binomial standard error)
    n = baseline['total']
    p_base = baseline['success_rate']
    p_rfl = rfl['success_rate']
    se_base = (p_base * (1 - p_base) / n) ** 0.5
    se_rfl = (p_rfl * (1 - p_rfl) / n) ** 0.5
    se_diff = (se_base**2 + se_rfl**2) ** 0.5
    
    print("STATISTICS:")
    print(f"  Baseline SE:     {se_base:.3f}")
    print(f"  RFL SE:          {se_rfl:.3f}")
    print(f"  Difference SE:   {se_diff:.3f}")
    print(f"  Z-score:         {delta_success / se_diff:.2f}" if se_diff > 0 else "  Z-score:         N/A")
    print()
    
    # Interpretation
    if abs(delta_success) < 0.05:
        print("INTERPRETATION: Case A - RFL ≈ Baseline (no clear uplift)")
        print("  This could be:")
        print("    - Noise (expected on 100 cycles)")
        print("    - Policy not yet affecting search behavior")
        print("    - Update rule too weak or features not discriminative")
    elif delta_success > 0.05:
        print("INTERPRETATION: Case C - RFL > Baseline (positive uplift!)")
    elif delta_success < -0.05:
        print("INTERPRETATION: Case B - RFL < Baseline (policy hurting)")
        print("  This means policy is doing something, just wrong direction.")
        print("  Consider: flipping signs, reducing step size, or adjusting update rule.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: compare_uplift.py <baseline.jsonl> <rfl.jsonl>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])

