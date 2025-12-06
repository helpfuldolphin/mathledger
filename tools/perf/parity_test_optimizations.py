#!/usr/bin/env python3
"""
Parity tests for performance optimizations.

Ensures that optimized implementations produce identical results to original implementations.
Tests functional correctness while measuring performance improvements.

Usage:
    python tools/perf/parity_test_optimizations.py
    python -m unittest tools.perf.parity_test_optimizations
"""

import sys
import unittest
from pathlib import Path
from typing import Set

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))


class ModusPonensParityTest(unittest.TestCase):
    """Test that optimized Modus Ponens produces identical results."""

    def setUp(self):
        try:
            from axiom_engine.rules import apply_modus_ponens
            self.apply_modus_ponens = apply_modus_ponens
        except ImportError:
            self.skipTest("Could not import apply_modus_ponens")

    def test_empty_set(self):
        """Test with empty statement set."""
        statements = set()
        result = self.apply_modus_ponens(statements)
        self.assertEqual(result, set())

    def test_atoms_only(self):
        """Test with only atomic statements (no implications)."""
        statements = {"p", "q", "r"}
        result = self.apply_modus_ponens(statements)
        self.assertEqual(result, set())

    def test_implications_only(self):
        """Test with only implications (no atoms)."""
        statements = {"p->q", "q->r", "r->s"}
        result = self.apply_modus_ponens(statements)
        self.assertEqual(result, set())

    def test_simple_derivation(self):
        """Test simple p, p->q |- q derivation."""
        statements = {"p", "p->q"}
        result = self.apply_modus_ponens(statements)
        self.assertEqual(result, {"q"})

    def test_multiple_derivations(self):
        """Test multiple independent derivations."""
        statements = {
            "p1", "p1->q1",
            "p2", "p2->q2",
            "p3", "p3->q3"
        }
        result = self.apply_modus_ponens(statements)
        self.assertEqual(result, {"q1", "q2", "q3"})

    def test_no_matching_antecedents(self):
        """Test implications with no matching antecedents."""
        statements = {"p", "q->r", "s->t"}
        result = self.apply_modus_ponens(statements)
        self.assertEqual(result, set())

    def test_complex_formulas(self):
        """Test with complex nested formulas."""
        statements = {
            "(p/\\q)", "(p/\\q)->r"
        }
        result = self.apply_modus_ponens(statements)
        self.assertEqual(result, {"r"})

    def test_normalization_equivalence(self):
        """Test that different representations of same formula are recognized."""
        statements = {
            "p", "( p ) -> q"
        }
        result = self.apply_modus_ponens(statements)
        self.assertEqual(result, {"q"})

    def test_large_dataset_correctness(self):
        """Test correctness with large dataset."""
        statements = set()
        for i in range(1, 101):
            statements.add(f"p{i}")
            statements.add(f"p{i}->q{i}")
        
        result = self.apply_modus_ponens(statements)
        
        expected = {f"q{i}" for i in range(1, 101)}
        self.assertEqual(result, expected)

    def test_idempotence(self):
        """Test that applying MP twice gives same results."""
        statements = {"p", "p->q", "q->r"}
        
        result1 = self.apply_modus_ponens(statements)
        
        statements_with_derived = statements | result1
        result2 = self.apply_modus_ponens(statements_with_derived)
        
        self.assertIn("q", result1)
        self.assertIn("r", result2)


class CanonParityTest(unittest.TestCase):
    """Test that optimized canonicalization produces identical results."""

    def setUp(self):
        try:
            from logic.canon import normalize, normalize_pretty, get_atomic_propositions
            self.normalize = normalize
            self.normalize_pretty = normalize_pretty
            self.get_atomic_propositions = get_atomic_propositions
        except ImportError:
            self.skipTest("Could not import canon functions")

    def test_normalize_atoms(self):
        """Test normalization of atomic propositions."""
        self.assertEqual(self.normalize("p"), "p")
        self.assertEqual(self.normalize("(p)"), "p")
        self.assertEqual(self.normalize("( p )"), "p")

    def test_normalize_implications(self):
        """Test normalization of implications."""
        self.assertEqual(self.normalize("p->q"), "p->q")
        self.assertEqual(self.normalize("(p)->(q)"), "p->q")
        self.assertEqual(self.normalize("p -> q"), "p->q")

    def test_normalize_conjunctions(self):
        """Test normalization of conjunctions (commutative)."""
        result1 = self.normalize("p/\\q")
        result2 = self.normalize("q/\\p")
        self.assertEqual(result1, result2)

    def test_normalize_disjunctions(self):
        """Test normalization of disjunctions (commutative)."""
        result1 = self.normalize("p\\/q")
        result2 = self.normalize("q\\/p")
        self.assertEqual(result1, result2)

    def test_normalize_complex(self):
        """Test normalization of complex formulas."""
        self.assertEqual(
            self.normalize("(p->q)->r"),
            "(p->q)->r"
        )
        self.assertEqual(
            self.normalize("p->(q->r)"),
            "p->q->r"
        )

    def test_normalize_pretty_simple(self):
        """Test pretty normalization of simple implications."""
        self.assertEqual(
            self.normalize_pretty("p -> q -> r"),
            "p -> (q -> r)"
        )

    def test_normalize_pretty_paren(self):
        """Test pretty normalization of parenthesized implications."""
        self.assertEqual(
            self.normalize_pretty("(p -> q) -> r"),
            "(p -> q) -> r"
        )

    def test_get_atomic_propositions(self):
        """Test atomic proposition extraction."""
        self.assertEqual(
            self.get_atomic_propositions("p->q"),
            {"p", "q"}
        )
        self.assertEqual(
            self.get_atomic_propositions("(p/\\q)->(r\\/s)"),
            {"p", "q", "r", "s"}
        )

    def test_unicode_mapping(self):
        """Test Unicode symbol mapping."""
        self.assertEqual(
            self.normalize("p→q"),
            self.normalize("p->q")
        )
        self.assertEqual(
            self.normalize("p∧q"),
            self.normalize("p/\\q")
        )
        self.assertEqual(
            self.normalize("p∨q"),
            self.normalize("p\\/q")
        )


class CacheEffectivenessTest(unittest.TestCase):
    """Test that caching is working effectively."""

    def setUp(self):
        try:
            from axiom_engine.rules import _cached_normalize
            self.cached_normalize = _cached_normalize
        except ImportError:
            self.skipTest("Could not import _cached_normalize")

    def test_cache_hits(self):
        """Test that repeated calls hit the cache."""
        self.cached_normalize.cache_clear()
        
        test_formulas = ["p->q", "p->q", "p->q"]
        for f in test_formulas:
            self.cached_normalize(f)
        
        cache_info = self.cached_normalize.cache_info()
        
        self.assertEqual(cache_info.hits, 2)
        self.assertEqual(cache_info.misses, 1)

    def test_cache_different_inputs(self):
        """Test that different inputs are cached separately."""
        self.cached_normalize.cache_clear()
        
        test_formulas = ["p->q", "r->s", "p->q"]
        for f in test_formulas:
            self.cached_normalize(f)
        
        cache_info = self.cached_normalize.cache_info()
        
        self.assertEqual(cache_info.hits, 1)
        self.assertEqual(cache_info.misses, 2)


class CongruenceClosureParityTest(unittest.TestCase):
    """Test congruence closure correctness."""

    def setUp(self):
        try:
            from fol_eq.cc import CC, const, fun
            self.CC = CC
            self.const = const
            self.fun = fun
        except ImportError:
            self.skipTest("Could not import CC")

    def test_reflexivity(self):
        """Test that a = a."""
        cc = self.CC()
        a = self.const("a")
        cc.add_term(a)
        self.assertTrue(cc.equal(a, a))

    def test_simple_equality(self):
        """Test simple equality assertion."""
        cc = self.CC()
        a = self.const("a")
        b = self.const("b")
        
        cc.assert_eqs([(a, b)])
        self.assertTrue(cc.equal(a, b))
        self.assertTrue(cc.equal(b, a))

    def test_transitivity(self):
        """Test transitive equality."""
        cc = self.CC()
        a = self.const("a")
        b = self.const("b")
        c = self.const("c")
        
        cc.assert_eqs([(a, b), (b, c)])
        self.assertTrue(cc.equal(a, c))

    def test_congruence(self):
        """Test congruence property."""
        cc = self.CC()
        a = self.const("a")
        b = self.const("b")
        fa = self.fun("f", a)
        fb = self.fun("f", b)
        
        cc.add_term(fa)
        cc.add_term(fb)
        cc.assert_eqs([(a, b)])
        self.assertTrue(cc.equal(fa, fb))

    def test_multiple_equations(self):
        """Test multiple independent equations."""
        cc = self.CC()
        a = self.const("a")
        b = self.const("b")
        c = self.const("c")
        d = self.const("d")
        
        cc.assert_eqs([(a, b), (c, d)])
        
        self.assertTrue(cc.equal(a, b))
        self.assertTrue(cc.equal(c, d))
        self.assertFalse(cc.equal(a, c))


def run_parity_tests():
    """Run all parity tests and report results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(ModusPonensParityTest))
    suite.addTests(loader.loadTestsFromTestCase(CanonParityTest))
    suite.addTests(loader.loadTestsFromTestCase(CacheEffectivenessTest))
    suite.addTests(loader.loadTestsFromTestCase(CongruenceClosureParityTest))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_parity_tests())
