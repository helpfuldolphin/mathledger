#!/usr/bin/env python3
"""
Performance harness for Modus Ponens optimization.

Measures wall-time and call counts for synthetic datasets of varying sizes
to demonstrate the O(n^2) -> O(n) performance improvement.
"""

import unittest
import time
import json
import sys
import os
from typing import Set, Dict, Any, List
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from backend.repro.determinism import deterministic_slug

class ModusPonensBenchmark(unittest.TestCase):
    """Benchmark harness for Modus Ponens performance testing."""

    def setUp(self):
        self.results = {}

    def generate_synthetic_dataset(self, size: int) -> Set[str]:
        """Generate synthetic dataset with atoms and implications."""
        statements = set()

        for i in range(1, size // 2 + 1):
            statements.add(f'p{i}')

        for i in range(1, size // 2 + 1):
            statements.add(f'p{i}->q{i}')

        if size > 10:
            statements.add('(p1/\\p2)->r1')
            statements.add('(q1\\/q2)->r2')
            statements.add('r1->r2')

        return statements

    def measure_performance(self, func, statements: Set[str], iterations: int = 10) -> Dict[str, Any]:
        """Measure wall-time and call counts for a function."""
        times = []

        for _ in range(iterations):
            start_time = time.perf_counter()
            result = func(statements)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        return {
            'wall_time_ms': sum(times) * 1000 / len(times),  # Average in milliseconds
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000,
            'iterations': iterations,
            'result_size': len(result) if result else 0,
            'input_size': len(statements)
        }

    def test_modus_ponens_performance_100_atoms(self):
        """Benchmark MP with 100 atoms."""
        try:
            from axiom_engine.rules import apply_modus_ponens

            statements = self.generate_synthetic_dataset(100)
            result = self.measure_performance(apply_modus_ponens, statements, iterations=50)

            self.results['mp_100_atoms'] = result
            print(f"MP 100 atoms: {result['wall_time_ms']:.4f}ms avg, {result['result_size']} derivations")

        except ImportError:
            self.skipTest("Could not import apply_modus_ponens")

    def test_modus_ponens_performance_1k_atoms(self):
        """Benchmark MP with 1000 atoms."""
        try:
            from axiom_engine.rules import apply_modus_ponens

            statements = self.generate_synthetic_dataset(1000)
            result = self.measure_performance(apply_modus_ponens, statements, iterations=10)

            self.results['mp_1k_atoms'] = result
            print(f"MP 1k atoms: {result['wall_time_ms']:.4f}ms avg, {result['result_size']} derivations")

        except ImportError:
            self.skipTest("Could not import apply_modus_ponens")

    def test_modus_ponens_performance_10k_atoms(self):
        """Benchmark MP with 10000 atoms."""
        try:
            from axiom_engine.rules import apply_modus_ponens

            statements = self.generate_synthetic_dataset(10000)
            result = self.measure_performance(apply_modus_ponens, statements, iterations=3)

            self.results['mp_10k_atoms'] = result
            print(f"MP 10k atoms: {result['wall_time_ms']:.4f}ms avg, {result['result_size']} derivations")

        except ImportError:
            self.skipTest("Could not import apply_modus_ponens")

    def test_normalization_cache_effectiveness(self):
        """Test LRU cache effectiveness for normalization."""
        try:
            from axiom_engine.rules import _cached_normalize

            test_formulas = [
                'p -> q', '(p) -> (q)', 'p->q',  # Same normalized form
                'r /\\ s', '(r) /\\ (s)', 'r/\\s',  # Same normalized form
                'a \\/ b', 'a\\/b', '(a \\/ b)'   # Same normalized form
            ] * 100  # Repeat to test cache hits

            start_time = time.perf_counter()
            results = [_cached_normalize(f) for f in test_formulas]
            end_time = time.perf_counter()

            cache_info = _cached_normalize.cache_info()

            self.results['normalization_cache'] = {
                'wall_time_ms': (end_time - start_time) * 1000,
                'cache_hits': cache_info.hits,
                'cache_misses': cache_info.misses,
                'hit_rate': cache_info.hits / (cache_info.hits + cache_info.misses) if cache_info.hits + cache_info.misses > 0 else 0,
                'formulas_processed': len(test_formulas),
                'unique_results': len(set(results))
            }

            print(f"Normalization cache: {cache_info.hits} hits, {cache_info.misses} misses, {self.results['normalization_cache']['hit_rate']:.2%} hit rate")

        except ImportError:
            self.skipTest("Could not import _cached_normalize")

    def tearDown(self):
        """Save benchmark results to JSON file."""
        if self.results:
            os.makedirs('artifacts/perf', exist_ok=True)
            canonical_results = json.dumps(self.results, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
            slug = deterministic_slug('modus_ponens_bench', canonical_results)
            filename = f'artifacts/perf/modus_ponens_bench_{slug}.json'

            with open(filename, 'w', encoding='ascii') as f:
                f.write(canonical_results)

            print(f"Benchmark results saved to {filename}")

def run_benchmarks():
    """CLI entry point for running benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description='Run Modus Ponens performance benchmarks')
    parser.add_argument('--target', default='modus_ponens', help='Target to benchmark')
    parser.add_argument('--iters', type=int, default=10, help='Number of iterations')
    parser.add_argument('--out', help='Output file for results')

    args = parser.parse_args()

    if args.target == 'modus_ponens':
        suite = unittest.TestLoader().loadTestsFromTestCase(ModusPonensBenchmark)
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    else:
        print(f"Unknown target: {args.target}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        run_benchmarks()
    else:
        unittest.main()
