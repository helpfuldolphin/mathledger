#!/usr/bin/env python3
"""
Profile the entire derivation engine to identify bottlenecks.

Usage:
    python tools/perf/profile_engine.py --size 5000
    python tools/perf/profile_engine.py --size 5000 --output /tmp/profile.stats
"""

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))


def profile_modus_ponens(size: int):
    """Profile Modus Ponens with specified dataset size."""
    from axiom_engine.rules import apply_modus_ponens
    
    statements = set()
    for i in range(1, size // 2 + 1):
        statements.add(f'p{i}')
        statements.add(f'p{i}->q{i}')
    
    result = apply_modus_ponens(statements)
    print(f'Derived {len(result)} statements from {len(statements)} input statements')
    return result


def profile_canonicalization(count: int):
    """Profile canonicalization with specified formula count."""
    from logic.canon import normalize
    
    formulas = []
    for i in range(count // 4):
        formulas.append(f"p{i}->q{i}")
        formulas.append(f"(p{i})->(q{i})")
        formulas.append(f"p{i}/\\q{i}")
        formulas.append(f"q{i}/\\p{i}")
    
    results = [normalize(f) for f in formulas[:count]]
    print(f'Normalized {len(results)} formulas')
    return results


def main():
    parser = argparse.ArgumentParser(description='Profile MathLedger derivation engine')
    parser.add_argument('--size', type=int, default=5000, help='Dataset size')
    parser.add_argument('--target', choices=['modus_ponens', 'canonicalization', 'both'], 
                       default='modus_ponens', help='What to profile')
    parser.add_argument('--output', help='Output file for profile stats')
    
    args = parser.parse_args()
    
    if args.output:
        profiler = cProfile.Profile()
        profiler.enable()
    
    if args.target in ['modus_ponens', 'both']:
        print(f"\n=== Profiling Modus Ponens (size={args.size}) ===")
        profile_modus_ponens(args.size)
    
    if args.target in ['canonicalization', 'both']:
        print(f"\n=== Profiling Canonicalization (count={args.size}) ===")
        profile_canonicalization(args.size)
    
    if args.output:
        profiler.disable()
        profiler.dump_stats(args.output)
        
        print(f"\n=== Profile Statistics (by total time) ===")
        p = pstats.Stats(args.output)
        p.strip_dirs()
        p.sort_stats('tottime')
        p.print_stats(30)
        
        print(f"\n=== Profile Statistics (by cumulative time) ===")
        p.sort_stats('cumulative')
        p.print_stats(30)


if __name__ == '__main__':
    main()
