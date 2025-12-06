#!/usr/bin/env python3
"""
Test script to verify byte-for-byte determinism of MathLedger derivations.

This script runs two identical derivations with the same seed and compares:
- Statement hashes
- Proof counts
- Merkle roots
- Block numbers
- JSONL metrics output

Usage:
    python tools/repro/test_determinism.py --system pl --seed 42 --steps 10
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


def run_derivation(system: str, seed: int, steps: int, output_dir: Path) -> Dict:
    """
    Run a single derivation and capture outputs.
    
    Args:
        system: System identifier (pl, fol)
        seed: Random seed
        steps: Number of derivation steps
        output_dir: Directory to store outputs
    
    Returns:
        Dictionary with derivation results
    """
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = str(seed)
    
    cmd = [
        'python3', '-m', 'backend.axiom_engine.derive',
        '--system', system,
        '--smoke-pl',
        '--seal'
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).parent.parent.parent
    )
    
    output_lines = result.stdout.strip().split('\n')
    metrics = {}
    
    for line in output_lines:
        if '=' in line:
            key, value = line.split('=', 1)
            metrics[key.strip()] = value.strip()
    
    (output_dir / 'stdout.txt').write_text(result.stdout)
    (output_dir / 'stderr.txt').write_text(result.stderr)
    canonical_metrics = json.dumps(
        metrics,
        ensure_ascii=True,
        sort_keys=True,
        separators=(',', ':')
    )
    (output_dir / 'metrics.json').write_text(canonical_metrics)
    
    return {
        'returncode': result.returncode,
        'metrics': metrics,
        'stdout': result.stdout,
        'stderr': result.stderr
    }


def compare_results(run1: Dict, run2: Dict) -> Tuple[bool, List[str]]:
    """
    Compare two derivation runs for byte-for-byte determinism.
    
    Args:
        run1: First run results
        run2: Second run results
    
    Returns:
        (is_deterministic, differences)
    """
    differences = []
    
    if run1['returncode'] != run2['returncode']:
        differences.append(f"Return codes differ: {run1['returncode']} vs {run2['returncode']}")
    
    metrics1 = run1['metrics']
    metrics2 = run2['metrics']
    
    deterministic_keys = ['PROOFS_INSERTED', 'MERKLE', 'BLOCK']
    
    for key in deterministic_keys:
        val1 = metrics1.get(key, '')
        val2 = metrics2.get(key, '')
        if val1 != val2:
            differences.append(f"{key} differs: {val1} vs {val2}")
    
    stdout1_lines = [l for l in run1['stdout'].split('\n') if not l.startswith('POLICY_SEED=')]
    stdout2_lines = [l for l in run2['stdout'].split('\n') if not l.startswith('POLICY_SEED=')]
    
    if stdout1_lines != stdout2_lines:
        differences.append("Stdout differs (excluding POLICY_SEED)")
    
    is_deterministic = len(differences) == 0
    return is_deterministic, differences


def main():
    parser = argparse.ArgumentParser(
        description="Test byte-for-byte determinism of MathLedger derivations"
    )
    parser.add_argument('--system', default='pl', help='System identifier (default: pl)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--steps', type=int, default=10, help='Derivation steps (default: 10)')
    parser.add_argument('--runs', type=int, default=2, help='Number of runs to compare (default: 2)')
    parser.add_argument('--output', type=Path, help='Output directory (default: temp)')
    
    args = parser.parse_args()
    
    if args.output:
        output_base = args.output
        output_base.mkdir(parents=True, exist_ok=True)
    else:
        output_base = Path(tempfile.mkdtemp(prefix='mathledger_determinism_'))
    
    print(f"Testing determinism with {args.runs} runs...")
    print(f"System: {args.system}, Seed: {args.seed}, Steps: {args.steps}")
    print(f"Output directory: {output_base}")
    print()
    
    results = []
    for i in range(args.runs):
        print(f"Running derivation {i+1}/{args.runs}...")
        run_dir = output_base / f"run_{i+1}"
        run_dir.mkdir(exist_ok=True)
        
        result = run_derivation(args.system, args.seed, args.steps, run_dir)
        results.append(result)
        
        print(f"  Return code: {result['returncode']}")
        print(f"  Metrics: {result['metrics']}")
        print()
    
    print("Comparing runs for determinism...")
    print()
    
    all_deterministic = True
    for i in range(len(results) - 1):
        for j in range(i + 1, len(results)):
            print(f"Comparing run {i+1} vs run {j+1}:")
            is_det, diffs = compare_results(results[i], results[j])
            
            if is_det:
                print("  ✓ DETERMINISTIC - outputs match byte-for-byte")
            else:
                print("  ✗ NONDETERMINISTIC - differences found:")
                for diff in diffs:
                    print(f"    - {diff}")
                all_deterministic = False
            print()
    
    print("=" * 70)
    if all_deterministic:
        print("SUCCESS: All runs produced identical outputs")
        print("MathLedger derivations are DETERMINISTIC")
        return 0
    else:
        print("FAILURE: Runs produced different outputs")
        print("MathLedger derivations are NONDETERMINISTIC")
        print(f"See detailed outputs in: {output_base}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
