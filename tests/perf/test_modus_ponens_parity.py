#!/usr/bin/env python3
"""
Functional parity tests for Modus Ponens optimization.

These tests verify that the optimized O(n) implementation produces identical
results to the original O(n²) implementation using synthetic test cases.
"""

import unittest
import sys
import os
from typing import Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

def _original_apply_modus_ponens(statements: Set[str]) -> Set[str]:
    """Original O(n²) implementation for comparison."""
    try:
        from normalization.canon import normalize
    except Exception:
        from backend.axiom_engine.rules import _strip_outer
        normalize = lambda x: _strip_outer(x).replace(" ", "")

    from backend.axiom_engine.rules import _is_implication, _parse_implication

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

class ModusPonensFunctionalParityTest(unittest.TestCase):
    """Test that optimized Modus Ponens maintains functional parity."""

    def test_basic_modus_ponens(self):
        """Test basic MP: p, p->q ⊢ q"""
        try:
            from axiom_engine.rules import apply_modus_ponens

            statements = {'p', 'p->q'}

            original_result = _original_apply_modus_ponens(statements)
            optimized_result = apply_modus_ponens(statements)

            self.assertEqual(original_result, optimized_result)
            self.assertIn('q', optimized_result)

        except ImportError:
            self.skipTest("Could not import apply_modus_ponens")

    def test_no_false_positives(self):
        """Test that MP doesn't derive invalid conclusions."""
        try:
            from axiom_engine.rules import apply_modus_ponens

            statements = {'p', 'q', 'r'}  # No implications

            original_result = _original_apply_modus_ponens(statements)
            optimized_result = apply_modus_ponens(statements)

            self.assertEqual(original_result, optimized_result)
            self.assertEqual(len(optimized_result), 0)

        except ImportError:
            self.skipTest("Could not import apply_modus_ponens")

    def test_multiple_implications(self):
        """Test MP with multiple implications."""
        try:
            from axiom_engine.rules import apply_modus_ponens

            statements = {'p', 'p->q', 'q->r', 'r->s'}

            original_result = _original_apply_modus_ponens(statements)
            optimized_result = apply_modus_ponens(statements)

            self.assertEqual(original_result, optimized_result)
            self.assertIn('q', optimized_result)

        except ImportError:
            self.skipTest("Could not import apply_modus_ponens")

    def test_complex_formulas(self):
        """Test MP with complex nested formulas."""
        try:
            from axiom_engine.rules import apply_modus_ponens

            statements = {'(p/\\q)', '(p/\\q)->r', '((p/\\q)->r)->s'}

            original_result = _original_apply_modus_ponens(statements)
            optimized_result = apply_modus_ponens(statements)

            self.assertEqual(original_result, optimized_result)

        except ImportError:
            self.skipTest("Could not import apply_modus_ponens")

    def test_normalization_edge_cases(self):
        """Test MP with formulas requiring normalization."""
        try:
            from axiom_engine.rules import apply_modus_ponens

            statements = {'p -> q', 'p', '(p) -> (q)', 'p->q'}  # Various spacing/parens

            original_result = _original_apply_modus_ponens(statements)
            optimized_result = apply_modus_ponens(statements)

            self.assertEqual(original_result, optimized_result)

        except ImportError:
            self.skipTest("Could not import apply_modus_ponens")

    def test_empty_set(self):
        """Test MP with empty statement set."""
        try:
            from axiom_engine.rules import apply_modus_ponens

            statements = set()

            original_result = _original_apply_modus_ponens(statements)
            optimized_result = apply_modus_ponens(statements)

            self.assertEqual(original_result, optimized_result)
            self.assertEqual(len(optimized_result), 0)

        except ImportError:
            self.skipTest("Could not import apply_modus_ponens")

    def test_malformed_implications(self):
        """Test MP with malformed implications."""
        try:
            from axiom_engine.rules import apply_modus_ponens

            statements = {'p', '->', 'p->', '->q', 'p->'}

            original_result = _original_apply_modus_ponens(statements)
            optimized_result = apply_modus_ponens(statements)

            self.assertEqual(original_result, optimized_result)

        except ImportError:
            self.skipTest("Could not import apply_modus_ponens")

    def test_large_dataset_parity(self):
        """Test MP parity with larger synthetic dataset."""
        try:
            from axiom_engine.rules import apply_modus_ponens

            statements = set()
            for i in range(1, 11):
                statements.add(f'p{i}')
                if i < 10:
                    statements.add(f'p{i}->p{i+1}')

            original_result = _original_apply_modus_ponens(statements)
            optimized_result = apply_modus_ponens(statements)

            self.assertEqual(original_result, optimized_result)

        except ImportError:
            self.skipTest("Could not import apply_modus_ponens")

    def test_unicode_formulas(self):
        """Test MP with Unicode logical symbols."""
        try:
            from axiom_engine.rules import apply_modus_ponens

            statements = {'p', 'p → q', 'q ∧ r', '(q ∧ r) → s'}

            original_result = _original_apply_modus_ponens(statements)
            optimized_result = apply_modus_ponens(statements)

            self.assertEqual(original_result, optimized_result)

        except ImportError:
            self.skipTest("Could not import apply_modus_ponens")

if __name__ == '__main__':
    unittest.main()
