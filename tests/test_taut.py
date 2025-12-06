"""
Tests for truth table tautology checker.
"""

import pytest
from backend.logic.taut import truth_table_is_tautology, _evaluate_formula, _extract_atoms


class TestTruthTableTautology:
    """Test the truth table tautology checker."""

    def test_simple_tautologies(self):
        """Test simple tautologies."""
        assert truth_table_is_tautology("p -> p")
        assert truth_table_is_tautology("p \\/ ~p")
        assert truth_table_is_tautology("~p \\/ p")

    def test_simple_non_tautologies(self):
        """Test simple non-tautologies."""
        assert not truth_table_is_tautology("p")
        assert not truth_table_is_tautology("p /\\ ~p")
        assert not truth_table_is_tautology("p -> q")

    def test_complex_tautologies(self):
        """Test complex tautologies."""
        assert truth_table_is_tautology("(p -> q) \\/ (q -> p)")
        assert truth_table_is_tautology("((p -> q) -> p) -> p")  # Peirce's law
        assert truth_table_is_tautology("p -> (q -> p)")
        assert truth_table_is_tautology("(p /\\ q) -> p")
        assert truth_table_is_tautology("(p /\\ q) -> q")

    def test_complex_non_tautologies(self):
        """Test complex non-tautologies."""
        assert not truth_table_is_tautology("p /\\ q")
        assert not truth_table_is_tautology("p -> q")
        assert not truth_table_is_tautology("(p -> q) /\\ (q -> p)")

    def test_tautologies_with_multiple_atoms(self):
        """Test tautologies with multiple atomic propositions."""
        assert truth_table_is_tautology("p -> (q -> p)")
        assert truth_table_is_tautology("(p -> q) -> ((q -> r) -> (p -> r))")
        assert truth_table_is_tautology("p -> (q -> (p /\\ q))")

    def test_extract_atoms(self):
        """Test atomic proposition extraction."""
        assert _extract_atoms("p") == ['p']
        assert set(_extract_atoms("p -> q")) == {'p', 'q'}
        assert set(_extract_atoms("p /\\ q /\\ r")) == {'p', 'q', 'r'}
        assert set(_extract_atoms("(p -> q) /\\ (r -> s)")) == {'p', 'q', 'r', 's'}

    def test_evaluate_formula_simple(self):
        """Test formula evaluation with simple cases."""
        truth_values = {'p': True, 'q': False}

        assert _evaluate_formula("p", truth_values) == True
        assert _evaluate_formula("q", truth_values) == False
        assert _evaluate_formula("p -> q", truth_values) == False  # True -> False = False
        assert _evaluate_formula("q -> p", truth_values) == True   # False -> True = True

    def test_evaluate_formula_complex(self):
        """Test formula evaluation with complex cases."""
        truth_values = {'p': True, 'q': False, 'r': True}

        # Test conjunctions
        assert _evaluate_formula("p /\\ q", truth_values) == False  # True /\\ False = False
        assert _evaluate_formula("p /\\ r", truth_values) == True   # True /\\ True = True

        # Test disjunctions
        assert _evaluate_formula("p \\/ q", truth_values) == True   # True \\/ False = True
        assert _evaluate_formula("q \\/ r", truth_values) == True   # False \\/ True = True

        # Test implications
        assert _evaluate_formula("p -> q", truth_values) == False   # True -> False = False
        assert _evaluate_formula("q -> p", truth_values) == True    # False -> True = True
        assert _evaluate_formula("p -> r", truth_values) == True    # True -> True = True

    def test_evaluate_formula_with_parentheses(self):
        """Test formula evaluation with parentheses."""
        truth_values = {'p': True, 'q': False, 'r': True}

        # Test parenthesized expressions
        assert _evaluate_formula("(p /\\ q)", truth_values) == False
        assert _evaluate_formula("(p \\/ q)", truth_values) == True
        assert _evaluate_formula("(p -> q)", truth_values) == False

        # Test nested parentheses
        assert _evaluate_formula("((p /\\ q) \\/ r)", truth_values) == True
        assert _evaluate_formula("(p -> (q /\\ r))", truth_values) == False

    def test_evaluate_formula_negation(self):
        """Test formula evaluation with negation."""
        truth_values = {'p': True, 'q': False}

        # Test negation
        assert _evaluate_formula("~p", truth_values) == False  # ~True = False
        assert _evaluate_formula("~q", truth_values) == True   # ~False = True

        # Test negation with other operators
        assert _evaluate_formula("~p /\\ q", truth_values) == False  # False /\\ False = False
        assert _evaluate_formula("~p \\/ q", truth_values) == False  # False \\/ False = False
        assert _evaluate_formula("p \\/ ~q", truth_values) == True   # True \\/ True = True

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty formula
        assert not truth_table_is_tautology("")

        # Single atom (not a tautology)
        assert not truth_table_is_tautology("p")

        # Contradiction
        assert not truth_table_is_tautology("p /\\ ~p")

        # True constant
        assert truth_table_is_tautology("p \\/ ~p")  # This is a tautology

    def test_tautology_with_three_atoms(self):
        """Test tautology detection with three atomic propositions."""
        # Test with p, q, r
        assert truth_table_is_tautology("p -> (q -> p)")
        assert truth_table_is_tautology("(p -> q) -> ((q -> r) -> (p -> r))")
        assert truth_table_is_tautology("p -> (q -> (p /\\ q))")

        # Non-tautologies
        assert not truth_table_is_tautology("p /\\ q /\\ r")
        assert not truth_table_is_tautology("(p -> q) /\\ (q -> r)")

    def test_tautology_performance(self):
        """Test that tautology checking doesn't take too long."""
        import time

        start_time = time.time()
        result = truth_table_is_tautology("((p -> q) -> p) -> p")
        end_time = time.time()

        # Should complete quickly (within 1 second)
        assert end_time - start_time < 1.0
        assert result == True  # Peirce's law is a tautology
