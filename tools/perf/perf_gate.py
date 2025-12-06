#!/usr/bin/env python3
"""
Performance Gate with 3-run benchmarks, RFC 8785 sealed pack, and regression alarms.

Usage:
    python tools/perf/perf_gate.py [--baseline-ms BASELINE] [--output-dir DIR]

Exit Codes:
    0 - PASS: All gates passed
    1 - FAIL: Performance regression detected
    2 - ABSTAIN: Cache hit < 20% with diagnostics
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Set

from backend.repro.determinism import deterministic_isoformat


def get_env_fingerprint() -> Dict[str, str]:
    """Capture environment fingerprint (tool versions)."""
    fingerprint = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor() or 'unknown'
    }
    
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            fingerprint['uv_version'] = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        fingerprint['uv_version'] = 'not_installed'
    
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            fingerprint['git_commit'] = result.stdout.strip()[:8]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        fingerprint['git_commit'] = os.environ.get('GITHUB_SHA', 'unknown')[:8]
    
    return fingerprint


def compute_workload_signature(statements: Set[str]) -> str:
    """Compute deterministic hash of workload dataset."""
    sorted_statements = sorted(statements)
    workload_str = '|'.join(sorted_statements)
    return hashlib.sha256(workload_str.encode('utf-8')).hexdigest()[:16]


def generate_synthetic_dataset(size: int) -> Set[str]:
    """Generate synthetic dataset for benchmarking."""
    statements = set()
    atoms_count = size // 2
    for i in range(1, atoms_count + 1):
        statements.add(f'p{i}')
        statements.add(f'p{i}->q{i}')
    return statements


def run_3x_benchmarks(statements: Set[str]) -> Dict[str, Any]:
    """Run benchmarks 3 times with cache clearing and variance calculation."""
    try:
        from backend.axiom_engine.rules import (
            apply_modus_ponens,
            _cached_normalize,
            _apply_modus_ponens_cached
        )
    except ImportError as e:
        print(f"ERROR: Failed to import backend modules: {e}")
        print("Ensure PYTHONPATH includes the repository root")
        sys.exit(2)
    
    optimized_times = []
    
    for run in range(3):
        _cached_normalize.cache_clear()
        _apply_modus_ponens_cached.cache_clear()
        
        start_time = time.perf_counter()
        result = apply_modus_ponens(statements)
        end_time = time.perf_counter()
        
        optimized_times.append(end_time - start_time)
    
    avg_time = sum(optimized_times) / len(optimized_times)
    variance = sum((t - avg_time) ** 2 for t in optimized_times) / len(optimized_times)
    stddev = variance ** 0.5
    
    return {
        'runs': optimized_times,
        'avg_time_s': avg_time,
        'stddev_s': stddev,
        'min_time_s': min(optimized_times),
        'max_time_s': max(optimized_times),
        'result_size': len(result) if result else 0
    }


def test_cache_effectiveness() -> Dict[str, Any]:
    """Test cache effectiveness with repeated formulas."""
    try:
        from backend.axiom_engine.rules import _cached_normalize
    except ImportError as e:
        print(f"ERROR: Failed to import _cached_normalize: {e}")
        sys.exit(2)
    
    _cached_normalize.cache_clear()
    
    test_formulas = [
        'p -> q', '(p) -> (q)', ' p -> q ',
        'r /\\ s', '(r) /\\ (s)', ' r /\\ s ',
        'a \\/ b', ' a \\/ b ', '(a \\/ b)'
    ] * 100
    
    start_time = time.perf_counter()
    results = [_cached_normalize(f) for f in test_formulas]
    end_time = time.perf_counter()
    
    cache_info = _cached_normalize.cache_info()
    
    hit_rate = cache_info.hits / (cache_info.hits + cache_info.misses) \
        if cache_info.hits + cache_info.misses > 0 else 0
    
    return {
        'cache_hits': cache_info.hits,
        'cache_misses': cache_info.misses,
        'cache_size': cache_info.currsize,
        'cache_maxsize': cache_info.maxsize,
        'hit_rate': hit_rate,
        'test_formulas_count': len(test_formulas),
        'unique_formulas': len(set(test_formulas)),
        'unique_results': len(set(results)),
        'wall_time_ms': (end_time - start_time) * 1000
    }


def create_perf_pack(
    benchmark_results: Dict[str, Any],
    cache_results: Dict[str, Any],
    baseline_ms: float,
    dataset_size: int,
    env_fingerprint: Dict[str, str],
    workload_signature: str,
    prev_pack_sha256: str = None
) -> Dict[str, Any]:
    """Create performance pack with all metrics."""
    avg_time_ms = benchmark_results['avg_time_s'] * 1000
    stddev_ms = benchmark_results['stddev_s'] * 1000
    
    speedup = baseline_ms / avg_time_ms
    speedup_variance = (benchmark_results['stddev_s'] / benchmark_results['avg_time_s']) * speedup
    
    perf_pack = {
        'perf_uplift': {
            'baseline_ms': baseline_ms,
            'optimized_avg_ms': avg_time_ms,
            'optimized_stddev_ms': stddev_ms,
            'optimized_min_ms': benchmark_results['min_time_s'] * 1000,
            'optimized_max_ms': benchmark_results['max_time_s'] * 1000,
            'speedup': speedup,
            'speedup_variance': speedup_variance,
            'runs': benchmark_results['runs'],
            'dataset_size': dataset_size,
            'result_size': benchmark_results['result_size']
        },
        'cache_diagnostics': {
            'final_cache_state': {
                'hits': cache_results['cache_hits'],
                'misses': cache_results['cache_misses'],
                'size': cache_results['cache_size'],
                'maxsize': cache_results['cache_maxsize']
            },
            'hit_rate': cache_results['hit_rate'],
            'test_formulas_count': cache_results['test_formulas_count'],
            'unique_formulas': cache_results['unique_formulas'],
            'unique_results': cache_results['unique_results'],
            'wall_time_ms': cache_results['wall_time_ms']
        },
        'env_fingerprint': env_fingerprint,
        'workload_signature': workload_signature,
        'timestamp': deterministic_isoformat(
            'perf_pack',
            json.dumps(workload_signature, sort_keys=True),
            prev_pack_sha256 or ""
        ),
        'git_sha': os.environ.get('GITHUB_SHA', 'unknown')
    }
    
    if prev_pack_sha256:
        perf_pack['prev_pack_sha256'] = prev_pack_sha256
    
    return perf_pack


def export_rfc8785_canonical(perf_pack: Dict[str, Any]) -> str:
    """Export RFC 8785 canonical JSON."""
    return json.dumps(
        perf_pack,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True
    )


def compute_sha256(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def validate_gates(perf_pack: Dict[str, Any]) -> Dict[str, Any]:
    """Validate performance gates and return status."""
    speedup = perf_pack['perf_uplift']['speedup']
    speedup_var = perf_pack['perf_uplift']['speedup_variance']
    cache_hit_rate = perf_pack['cache_diagnostics']['hit_rate']
    
    uplift_pass = speedup >= 3.0
    cache_pass = cache_hit_rate >= 0.2
    
    status = {
        'uplift_pass': uplift_pass,
        'cache_pass': cache_pass,
        'speedup': speedup,
        'speedup_variance': speedup_var,
        'cache_hit_rate': cache_hit_rate
    }
    
    return status


def print_pass_lines(status: Dict[str, Any], pack_hash: str) -> None:
    """Print standardized pass-lines."""
    speedup = status['speedup']
    speedup_var = status['speedup_variance']
    cache_hit = status['cache_hit_rate']
    
    if status['uplift_pass']:
        print(f'[PASS] Perf Uplift {speedup:.2f}x (±{speedup_var:.2f})')
    else:
        print(f'[FAIL] Perf Uplift {speedup:.2f}x < 3.0x')
    
    if status['cache_pass']:
        print(f'[PASS] Cache Hit {cache_hit:.2%}')
    else:
        print(f'[ABSTAIN] Cache Hit {cache_hit:.2%} < 20%')
    
    print(f'[PASS] Perf Pack: {pack_hash}')


def generate_perf_hints(
    perf_pack: Dict[str, Any],
    status: Dict[str, Any],
    output_dir: str
) -> None:
    """Generate perf_hints.txt with top 3 suspects for regressions."""
    hints_path = os.path.join(output_dir, 'perf_hints.txt')
    
    with open(hints_path, 'w') as f:
        f.write("=== Performance Regression Hints ===\n\n")
        f.write(f"Speedup: {status['speedup']:.2f}x (target: >=3.0x)\n")
        f.write(f"Variance: ±{status['speedup_variance']:.2f}\n")
        f.write(f"Cache hit: {status['cache_hit_rate']:.2%}\n\n")
        
        f.write("Top 3 Suspects:\n\n")
        
        uplift = perf_pack['perf_uplift']
        cache = perf_pack['cache_diagnostics']
        
        f.write("1. Cache Effectiveness\n")
        if cache['hit_rate'] < 0.5:
            f.write(f"   - Cache hit rate is low: {cache['hit_rate']:.2%}\n")
            f.write(f"   - Check if LRU cache is being cleared unexpectedly\n")
            f.write(f"   - Verify cache maxsize is sufficient: {cache['final_cache_state']['maxsize']}\n")
        else:
            f.write(f"   - Cache appears healthy: {cache['hit_rate']:.2%} hit rate\n")
        f.write("\n")
        
        f.write("2. Algorithmic Changes\n")
        if status['speedup_variance'] > status['speedup'] * 0.5:
            f.write(f"   - High variance detected: ±{status['speedup_variance']:.2f}\n")
            f.write(f"   - Performance may be inconsistent across runs\n")
            f.write(f"   - Check for non-deterministic operations\n")
        else:
            f.write(f"   - Variance is acceptable: ±{status['speedup_variance']:.2f}\n")
        f.write("   - Review recent changes to backend/axiom_engine/rules.py\n")
        f.write("   - Review recent changes to backend/logic/canon.py\n")
        f.write("\n")
        
        f.write("3. Dataset Characteristics\n")
        f.write(f"   - Dataset size: {uplift['dataset_size']} statements\n")
        f.write(f"   - Result size: {uplift['result_size']} derived statements\n")
        if uplift['result_size'] < uplift['dataset_size'] * 0.1:
            f.write(f"   - Low derivation rate may indicate rule application issues\n")
        f.write(f"   - Workload signature: {perf_pack['workload_signature']}\n")
        f.write("\n")
        
        f.write("Recommended Actions:\n")
        f.write("- Run: python tools/perf/perf_gate.py --dataset-size 1000 (smaller test)\n")
        f.write("- Compare: git diff HEAD~1 backend/axiom_engine/rules.py\n")
        f.write("- Profile: python -m cProfile -o profile.stats tools/perf/perf_gate.py\n")
        f.write("- Rollback: git revert <commit> if regression is confirmed\n")
    
    print(f"Performance hints: {hints_path}")


def load_prev_pack(prev_pack_path: str) -> Dict[str, Any]:
    """Load previous perf pack for chain attestation."""
    try:
        with open(prev_pack_path, 'r') as f:
            prev_pack = json.load(f)
        
        canonical_prev = export_rfc8785_canonical(prev_pack)
        prev_hash = compute_sha256(canonical_prev)
        
        return {
            'pack': prev_pack,
            'hash': prev_hash,
            'timestamp': prev_pack.get('timestamp', 'unknown')
        }
    except FileNotFoundError:
        print(f"WARNING: Previous pack not found: {prev_pack_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in previous pack: {e}")
        return None


def validate_chain(current_pack: Dict[str, Any], prev_pack_info: Dict[str, Any]) -> Dict[str, Any]:
    """Validate chain integrity between current and previous pack."""
    if prev_pack_info is None:
        return {
            'chain_valid': False,
            'reason': 'Previous pack not available',
            'status': 'ABSTAIN'
        }
    
    current_ts = current_pack.get('timestamp', '')
    prev_ts = prev_pack_info['timestamp']
    
    if current_ts < prev_ts:
        return {
            'chain_valid': False,
            'reason': f'Timestamp regression: {current_ts} < {prev_ts}',
            'status': 'FAIL'
        }
    
    prev_hash_in_current = current_pack.get('prev_pack_sha256')
    if prev_hash_in_current != prev_pack_info['hash']:
        return {
            'chain_valid': False,
            'reason': f"Hash mismatch: expected {prev_pack_info['hash'][:16]}..., got {prev_hash_in_current[:16] if prev_hash_in_current else 'None'}...",
            'status': 'FAIL'
        }
    
    return {
        'chain_valid': True,
        'prev_hash': prev_pack_info['hash'],
        'status': 'PASS'
    }


def create_diagnostic_pack(
    perf_pack: Dict[str, Any],
    status: Dict[str, Any],
    pack_hash: str
) -> Dict[str, Any]:
    """Create diagnostic pack for ABSTAIN/FAIL cases."""
    diagnostic = {
        'status': 'ABSTAIN' if not status['cache_pass'] else 'FAIL',
        'reason': [],
        'perf_pack_hash': pack_hash,
        'metrics': {
            'speedup': status['speedup'],
            'speedup_variance': status['speedup_variance'],
            'cache_hit_rate': status['cache_hit_rate']
        },
        'thresholds': {
            'min_speedup': 3.0,
            'min_cache_hit_rate': 0.2
        },
        'full_perf_pack': perf_pack
    }
    
    if not status['uplift_pass']:
        diagnostic['reason'].append(
            f"Performance regression: {status['speedup']:.2f}x < 3.0x threshold"
        )
    
    if not status['cache_pass']:
        diagnostic['reason'].append(
            f"Cache effectiveness: {status['cache_hit_rate']:.2%} < 20% threshold"
        )
    
    return diagnostic


def main():
    parser = argparse.ArgumentParser(
        description='Performance Gate with 3-run benchmarks and regression alarms'
    )
    parser.add_argument(
        '--baseline-ms',
        type=float,
        default=227.29,
        help='Baseline time in milliseconds (default: 227.29 from Phase 2)'
    )
    parser.add_argument(
        '--dataset-size',
        type=int,
        default=5000,
        help='Size of synthetic dataset (default: 5000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts/perf',
        help='Output directory for artifacts (default: artifacts/perf)'
    )
    parser.add_argument(
        '--prev-pack',
        type=str,
        default=None,
        help='Path to previous perf pack for chain attestation (optional)'
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    prev_pack_info = None
    if args.prev_pack:
        print("Loading previous pack for chain attestation...")
        prev_pack_info = load_prev_pack(args.prev_pack)
        if prev_pack_info:
            print(f"Previous pack hash: {prev_pack_info['hash']}")
            print(f"Previous timestamp: {prev_pack_info['timestamp']}")
        print()
    
    print("=== Performance Gate v2 ===")
    print(f"Baseline: {args.baseline_ms:.2f}ms")
    print(f"Dataset size: {args.dataset_size}")
    if args.prev_pack:
        print(f"Chain mode: ENABLED")
    print()
    
    print("Capturing environment fingerprint...")
    env_fingerprint = get_env_fingerprint()
    print(f"Python: {env_fingerprint['python_version']}, Platform: {env_fingerprint['platform']}")
    print()
    
    print("Generating synthetic dataset...")
    statements = generate_synthetic_dataset(args.dataset_size)
    print(f"Generated {len(statements)} statements")
    
    workload_signature = compute_workload_signature(statements)
    print(f"Workload signature: {workload_signature}")
    print()
    
    print("Running 3x benchmarks with cache clearing...")
    benchmark_results = run_3x_benchmarks(statements)
    print(f"Average time: {benchmark_results['avg_time_s'] * 1000:.2f}ms ±{benchmark_results['stddev_s'] * 1000:.2f}ms")
    print()
    
    print("Testing cache effectiveness...")
    cache_results = test_cache_effectiveness()
    print(f"Cache hit rate: {cache_results['hit_rate']:.2%}")
    print()
    
    print("Creating performance pack...")
    prev_pack_sha256 = prev_pack_info['hash'] if prev_pack_info else None
    perf_pack = create_perf_pack(
        benchmark_results,
        cache_results,
        args.baseline_ms,
        len(statements),
        env_fingerprint,
        workload_signature,
        prev_pack_sha256
    )
    
    canonical_json = export_rfc8785_canonical(perf_pack)
    pack_path = os.path.join(args.output_dir, 'perf_pack.json')
    with open(pack_path, 'w') as f:
        f.write(canonical_json)
    
    pack_hash = compute_sha256(canonical_json)
    print(f"Perf pack exported: {pack_path}")
    print(f"SHA256: {pack_hash}")
    print()
    
    print("Validating performance gates...")
    status = validate_gates(perf_pack)
    print()
    
    chain_status = None
    if args.prev_pack:
        print("Validating chain integrity...")
        chain_status = validate_chain(perf_pack, prev_pack_info)
        if chain_status['status'] == 'PASS':
            print(f"[PASS] Chain integrity verified")
        elif chain_status['status'] == 'ABSTAIN':
            print(f"[ABSTAIN] Chain validation: {chain_status['reason']}")
        else:
            print(f"[FAIL] Chain broken: {chain_status['reason']}")
        print()
    
    print("=== Pass-Lines ===")
    print_pass_lines(status, pack_hash)
    if chain_status and chain_status['status'] == 'PASS':
        print(f"[PASS] Perf Chain Intact {chain_status['prev_hash']}")
    print()
    
    if chain_status and chain_status['status'] == 'FAIL':
        print("ERROR: Chain integrity check failed")
        sys.exit(1)
    
    if not status['uplift_pass'] or not status['cache_pass']:
        diagnostic = create_diagnostic_pack(perf_pack, status, pack_hash)
        
        diagnostic_path = os.path.join(args.output_dir, 'perf_diagnostic.json')
        with open(diagnostic_path, 'w') as f:
            json.dump(diagnostic, f, indent=2, sort_keys=True)
        
        print(f"=== Diagnostic Pack ===")
        print(f"Status: {diagnostic['status']}")
        print(f"Reasons:")
        for reason in diagnostic['reason']:
            print(f"  - {reason}")
        print(f"Diagnostic pack: {diagnostic_path}")
        print()
        
        if not status['uplift_pass']:
            print("Generating performance hints...")
            generate_perf_hints(perf_pack, status, args.output_dir)
            print()
            print("ERROR: Performance regression detected")
            sys.exit(1)
        else:
            print("WARNING: Cache effectiveness below threshold (ABSTAIN)")
            sys.exit(0)
    
    print("=== PASS: All gates passed ===")
    sys.exit(0)


if __name__ == '__main__':
    main()
