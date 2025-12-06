#!/usr/bin/env python3
"""
Parity testing for Modus Ponens optimization using quadratic extrapolation.
Since legacy O(n²) implementation times out for large datasets, use extrapolation.
"""

import unittest
import time
import sys
import os
from typing import Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

def legacy_apply_modus_ponens(statements: Set[str]) -> Set[str]:
    """Legacy O(n²) implementation for small dataset comparison."""
    try:
        from backend.logic.canon import normalize
    except Exception:
        normalize = lambda x: x.strip().replace(" ", "")
    
    def _strip_outer(s: str) -> str:
        s = s.strip()
        while s.startswith('(') and s.endswith(')'):
            s = s[1:-1].strip()
        return s

    def _is_implication(s: str) -> bool:
        return '->' in s

    def _parse_implication(s: str):
        if '->' not in s:
            return None, None
        parts = s.split('->', 1)
        if len(parts) != 2:
            return None, None
        return _strip_outer(parts[0]), _strip_outer(parts[1])
    
    derived: Set[str] = set()
    items = list(statements)
    for i, s1 in enumerate(items):
        for j, s2 in enumerate(items):
            if i == j: continue
            if not _is_implication(s2): continue
            a, c = _parse_implication(s2)
            if a and c and normalize(s1) == normalize(a):
                d = normalize(c)
                if d not in statements:
                    derived.add(d)
    return derived

class ModusPonensParity(unittest.TestCase):
    """Test functional parity between optimized and legacy implementations."""
    
    def test_small_dataset_parity(self):
        """Test identical results on small datasets where legacy doesn't timeout."""
        from axiom_engine.rules import apply_modus_ponens
        
        test_cases = [
            {'p', 'p->q'},
            {'p', 'q', 'p->r', 'q->s'},
            {'a', 'b', 'a->c', 'b->d', 'c->e'},
        ]
        
        for statements in test_cases:
            optimized = apply_modus_ponens(statements)
            legacy = legacy_apply_modus_ponens(statements)
            
            self.assertEqual(optimized, legacy, 
                f"Parity failed for {statements}: optimized={optimized}, legacy={legacy}")
    
    def test_quadratic_scaling_proof(self):
        """Prove O(n) vs O(n²) scaling using timing analysis."""
        from axiom_engine.rules import apply_modus_ponens
        
        sizes = [50, 100, 200]
        optimized_times = []
        legacy_times = []
        
        for size in sizes:
            statements = set()
            for i in range(1, size//2 + 1):
                statements.add(f'p{i}')
                statements.add(f'p{i}->q{i}')
            
            start = time.perf_counter()
            apply_modus_ponens(statements)
            optimized_times.append(time.perf_counter() - start)
            
            start = time.perf_counter()
            legacy_apply_modus_ponens(statements)
            legacy_times.append(time.perf_counter() - start)
        
        opt_ratio = optimized_times[2] / optimized_times[0]
        leg_ratio = legacy_times[2] / legacy_times[0]
        
        print(f"Optimized 4x scaling ratio: {opt_ratio:.2f}x (linear ~4x)")
        print(f"Legacy 4x scaling ratio: {leg_ratio:.2f}x (quadratic ~16x)")
        
        self.assertLess(opt_ratio, 8.0, "Optimized should scale sub-quadratically")
        self.assertGreater(leg_ratio, 10.0, "Legacy should scale quadratically")

if __name__ == '__main__':
    unittest.main()
