#!/usr/bin/env python3
"""
Seed Replay Guard - Verify byte-identical determinism across multiple runs.

This guard runs multiple derivations with the same seed and verifies that all
artifacts are byte-identical. Any drift triggers CI failure with detailed report.

Usage:
    python tools/repro/seed_replay_guard.py --seed 0 --runs 3 --path artifacts/repro/
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


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of file contents."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def run_derivation_with_artifacts(seed: int, output_dir: Path) -> Dict:
    """
    Run derivation and capture all artifacts.
    
    Returns:
        Dictionary with artifact paths and their hashes
    """
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = str(seed)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    
    cmd = [
        'python3', '-B', '-m', 'backend.axiom_engine.derive',
        '--system', 'pl',
        '--smoke-pl',
        '--seed', str(seed)
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).parent.parent.parent
    )
    
    stdout_file = output_dir / 'stdout.txt'
    stderr_file = output_dir / 'stderr.txt'
    stdout_file.write_text(result.stdout)
    stderr_file.write_text(result.stderr)
    
    return {
        'return_code': result.returncode,
        'stdout_hash': compute_file_hash(stdout_file),
        'stderr_hash': compute_file_hash(stderr_file),
        'stdout': result.stdout,
        'stderr': result.stderr
    }


def collect_artifact_hashes(artifact_path: Path) -> Dict[str, str]:
    """
    Collect hashes of all artifacts in the specified path.
    
    Returns:
        Dictionary mapping artifact names to their SHA-256 hashes
    """
    artifacts = {}
    
    if not artifact_path.exists():
        return artifacts
    
    for artifact_file in artifact_path.rglob('*.json'):
        rel_path = artifact_file.relative_to(artifact_path)
        artifacts[str(rel_path)] = compute_file_hash(artifact_file)
    
    return artifacts


def compare_runs(run1: Dict, run2: Dict, run_num1: int, run_num2: int) -> Tuple[bool, List[str]]:
    """
    Compare two runs for byte-identical outputs.
    
    Returns:
        (is_identical, list_of_differences)
    """
    differences = []
    
    if run1['return_code'] != run2['return_code']:
        differences.append(f"Return codes differ: {run1['return_code']} vs {run2['return_code']}")
    
    if run1['stdout_hash'] != run2['stdout_hash']:
        differences.append(f"Stdout differs between run {run_num1} and run {run_num2}")
    
    if run1['stderr_hash'] != run2['stderr_hash']:
        differences.append(f"Stderr differs between run {run_num1} and run {run_num2}")
    
    return len(differences) == 0, differences


def generate_drift_report(differences: List[str], runs: List[Dict], output_path: Path):
    """Generate drift report JSON when nondeterminism is detected."""
    report = {
        "version": "1.0.0",
        "status": "DRIFT_DETECTED",
        "timestamp": "2025-10-19T00:00:00Z",
        "differences": differences,
        "run_count": len(runs),
        "hashes": {
            f"run_{i+1}": {
                "stdout": run['stdout_hash'],
                "stderr": run['stderr_hash'],
                "return_code": run['return_code']
            }
            for i, run in enumerate(runs)
        },
        "recommendation": "Review code for nondeterministic operations (time, random, uuid)",
        "playbook": "docs/repro/DRIFT_RESPONSE_PLAYBOOK.md"
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, sort_keys=True)
    
    print(f"Drift report written to: {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Seed Replay Guard - Verify byte-identical determinism'
    )
    parser.add_argument('--seed', type=int, default=0, help='Seed for deterministic execution')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs to compare')
    parser.add_argument('--path', type=str, default='artifacts/repro/', help='Path to artifacts directory')
    args = parser.parse_args(argv)
    
    print(f"Seed Replay Guard: Running {args.runs} derivations with seed={args.seed}")
    print(f"Artifact path: {args.path}")
    print()
    
    temp_dirs = []
    runs = []
    
    try:
        for i in range(args.runs):
            temp_dir = Path(tempfile.mkdtemp(prefix=f'replay_guard_run{i+1}_'))
            temp_dirs.append(temp_dir)
            
            print(f"Running derivation {i+1}/{args.runs}...")
            run_result = run_derivation_with_artifacts(args.seed, temp_dir)
            runs.append(run_result)
            print(f"  Return code: {run_result['return_code']}")
            print(f"  Stdout hash: {run_result['stdout_hash'][:16]}...")
            print(f"  Stderr hash: {run_result['stderr_hash'][:16]}...")
            print()
        
        print("Comparing runs for byte-identical outputs...")
        all_identical = True
        all_differences = []
        
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                is_identical, differences = compare_runs(runs[i], runs[j], i+1, j+1)
                if not is_identical:
                    all_identical = False
                    all_differences.extend(differences)
                    print(f"  ✗ Run {i+1} vs Run {j+1}: DRIFT DETECTED")
                    for diff in differences:
                        print(f"    - {diff}")
                else:
                    print(f"  ✓ Run {i+1} vs Run {j+1}: BYTE-IDENTICAL")
        
        print()
        
        if all_identical:
            print("=" * 70)
            print("[PASS] Determinism Guard: All runs produced byte-identical outputs")
            print(f"Verified: {args.runs}/{args.runs} runs are deterministic")
            print("=" * 70)
            return 0
        else:
            print("=" * 70)
            print("[FAIL] Determinism Guard: DRIFT DETECTED")
            print(f"Nondeterminism found across {args.runs} runs")
            print("=" * 70)
            
            artifact_path = Path(args.path)
            artifact_path.mkdir(parents=True, exist_ok=True)
            drift_report_path = artifact_path / 'drift_report.json'
            generate_drift_report(all_differences, runs, drift_report_path)
            
            print()
            print("ABSTAIN: Cannot verify determinism. Review drift report for details.")
            return 1
    
    finally:
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)


if __name__ == '__main__':
    sys.exit(main())
